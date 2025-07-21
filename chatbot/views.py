
from django.views.generic import TemplateView
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import os
from django.utils import timezone
from django.conf import settings
from .models import ChatSession, ChatMessage
from django.contrib.auth.mixins import LoginRequiredMixin

settings.GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class ChatbotView(TemplateView, LoginRequiredMixin):
    template_name = "chatbot.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['theme'] = self.request.GET.get('theme', 'Dark')
        context['file_uploaded'] = bool(self.request.session.get('faiss_index_path'))
        context['uploaded_filename'] = self.request.session.get('original_filename', '')

        chat_session_id = self.request.session.get('chat_session_id')
        if chat_session_id:
            context['chat_history'] = ChatMessage.objects.filter(
                session_id=chat_session_id
            ).order_by('created_at')
        else:
            context['chat_history'] = []

        return context

    def post(self, request):
        context = self.get_context_data()
        try:
            if 'pdf_file' in request.FILES:
                self._handle_file_upload(request)
                context['file_uploaded'] = True
                context['uploaded_filename'] = request.session.get('original_filename', '')
                request.session['chat_session_id'] = None  # reset session on new file
                request.session.modified = True

            if 'user_query' in request.POST:
                user_query = request.POST['user_query']

                # Get or create ChatSession
                session_key = request.session.session_key or request.session.save()
                chat_session, created = ChatSession.objects.get_or_create(
                    session_key=session_key,
                    pdf_file=request.session.get('original_filename', 'unknown.pdf'),
                    defaults={'user': request.user if request.user.is_authenticated else None}
                )
                request.session['chat_session_id'] = chat_session.id

                # Save user message
                ChatMessage.objects.create(
                    session=chat_session,
                    sender='user',
                    message=user_query
                )

                # Get bot response
                bot_response = self._handle_user_query(request)

                # Save bot message
                ChatMessage.objects.create(
                    session=chat_session,
                    sender='bot',
                    message=bot_response
                )

                context['chat_history'] = ChatMessage.objects.filter(session=chat_session)

        except Exception as e:
            context['error'] = str(e)

        return self.render_to_response(context)
    
    def _handle_file_upload(self, request):
        index_name = f"faiss_index_{uuid.uuid4().hex}"
        index_path = os.path.join("vectorstores", index_name)

        # Create directories in one call
        os.makedirs("vectorstores", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)

        # Process file upload
        file = request.FILES['pdf_file']
        file_path = os.path.join("uploads", file.name)

        # Use more efficient file writing
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks(chunk_size=65536):  # Larger chunk size
                destination.write(chunk)

        # Optimized PDF processing
        text = ""
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            # Process pages in parallel if large PDF
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                texts = list(executor.map(
                    lambda page: page.extract_text() or "",
                    pdf_reader.pages
                ))
            text = "".join(texts)

        if not text.strip():
            raise ValueError("PDF text extraction failed - no text found")

        # Split text with optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Slightly larger chunks
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Cache embeddings model
        if not hasattr(self, '_embeddings'):
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

        # Create and save vector store with progress indication
        vector_store = FAISS.from_texts(chunks, self._embeddings)
        vector_store.save_local(index_path)

        # Store session data
        request.session['faiss_index_path'] = index_path
        request.session['original_filename'] = file.name

    def _handle_user_query(self, request):
        index_path = request.session.get('faiss_index_path')
        if not index_path or not os.path.exists(index_path):
            raise ValueError("Please upload a PDF file first")

        # Use cached embeddings
        if not hasattr(self, '_embeddings'):
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

        # Load vector store with caching
        if not hasattr(request, '_vector_store'):
            request._vector_store = FAISS.load_local(
                index_path, 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
        vector_store = request._vector_store

        # Process query
        query = request.POST['user_query']

        # First try to get quick answer from vector store
        docs = vector_store.similarity_search(query, k=3)

        # Cache LLM instance
        if not hasattr(self, '_llm'):
            self._llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0.7
            )
            self._chain = load_qa_chain(self._llm, chain_type="stuff")

        # Get response
        result = self._chain({"input_documents": docs, "question": query})

        return result['output_text']

