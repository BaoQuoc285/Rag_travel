"""FastAPI app creation, logger configuration and main API routes."""
import sys
import os
import logging
import uuid
from fastapi import Cookie, Response
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from dotenv import load_dotenv
from src.components.vector_store.vector_database import QdrantRAG
from src.components.llm.gemini_model import GeminiModel
from src.components.rerank.ranking import load_compression_retriever, query_vector_database
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.components.store_history.history import get_session_history
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime
from src.components.store_history.history import clear_chat_history
from langchain.schema import SystemMessage, HumanMessage  # Add this import
from src.components.hyde.hypothetical import HyDe
from typing import Any, List
from src.components.router.semantic_router import SemanticRouter, QueryType
import httpx  # Add this import
import base64  # Add this import
from fastapi import Depends  # Add this import
from typing import Optional  # Add this import
from typing import Tuple  # Add this import
from src.components.store_history.history import (
    get_session_history, 
    clear_chat_history,
    update_session_context,  # Add this import
    get_session_context     # Add this import
)
from langchain_community.tools import TavilySearchResults  # Add this import
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qdrant_api = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

class CollectionRequest(BaseModel):  # Add this class
    collection: str 

# Add global variable to track current collection
current_collection = "my_documents"

def load_gemini():
    chat_prompt = (
        "You are a passionate Saigon tour guide sharing your knowledge and experiences. "
        "Use the following context to give personalized recommendations and insights. "
        "Share interesting stories and local perspectives when relevant. "
        "If you're not sure about something, suggest alternative places or activities instead. "
        "Keep the conversation natural and engaging."
        "\n\n"
        "{context}"
    )
    retriever_prompt = (
        "As a tour guide, consider the chat history and current question to understand "
        "what specific information about Saigon would be most helpful. "
        "Reformulate the question to find the most relevant details from our city guide database."
    )
    return GeminiModel(chat_prompt=chat_prompt, retriever_prompt=retriever_prompt)

@app.get("/")
async def read_root(request: Request, response: Response):
    # Generate new session_id if not exists
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    else:
        # Clear existing chat history when page is refreshed
        deleted_count = clear_chat_history(session_id)
        logger.info(f"Cleared {deleted_count} messages from chat history for session {session_id}")
    
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    logger.info(f"Session ID: {session_id}")
    return templates.TemplateResponse("index.html", {"request": request})

def analyze_query(question: str, lang: str) -> Tuple[bool, str]:
    """Analyze query quality and return if it needs more details"""
    words = [w for w in question.strip().split() if len(w) > 1]
    
    # If it's a location name only, store it as context but ask for more details
    if len(words) == 1 and any(loc in question.lower() for loc in ["quận", "phường", "đường"]):
        if lang == 'vi':
            return True, "Bạn muốn biết thông tin gì về khu vực này? Ví dụ: địa điểm du lịch, ăn uống, mua sắm, di tích lịch sử..."
        else:
            return True, "What would you like to know about this area? For example: tourist spots, food, shopping, historical sites..."
    
    return False, ""

@app.post("/query")
async def query(
    request: QueryRequest, 
    session_id: str = Cookie(default=None),  # Changed from None to default=None
    collection: Optional[str] = None  # Add optional collection parameter
):
    global current_collection
    # Ensure we have a session ID
    if not session_id:
        session_id = str(uuid.uuid4())
        response = StreamingResponse(
            iter(["Xin lỗi, vui lòng tải lại trang để bắt đầu phiên mới."]),
            media_type="text/plain"
        )
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        return response

    # Use specified collection or fall back to current_collection
    collection_name = collection or current_collection
    logger.info(f"Using collection: {collection_name}")

    # Get chat history for the session
    message_history = get_session_history(session_id)
    
    # Initialize fresh history if needed
    if session_id and not hasattr(message_history, 'messages'):
        logger.info(f"Initializing new chat history for session {session_id}")
        message_history = ChatMessageHistory()

    question = request.question
    #if not text or number remove all
    question = ''.join(e for e in question if e.isalnum() or e.isspace())   
    lang = llm.detect_language(question)
    logger.info(f"Detected language: {lang} for question: {question}")
    #print lang để kiểm tra ngôn ngữ
    print(lang)
    #lower
    question = question.lower()
    # hcm replace by hồ chí minh,tp replace by thành phố
    question = question.replace("hcm", "hồ chí minh").replace("tp", "thành phố")
    # q number replace by quận number
    question = question.replace("q1", "quận 1").replace("q2", "quận 2").replace("q3", "quận 3").replace("q4", "quận 4").replace("q5", "quận 5").replace("q6", "quận 6").replace("q7", "quận 7").replace("q8", "quận 8").replace("q9", "quận 9").replace("q10", "quận 10").replace("q11", "quận 11").replace("q12", "quận 12")
    # replace by đường
    question = question.replace("đg", "đường")
    # replace by phường
    question = question.replace("phg", "phường")
    # replace by noi tieng
    question = question.replace("noi tieng", "nổi tiếng")
    
    log_prefix = "📝 Nhận câu hỏi" if lang == 'vi' else "📝 Received question"
    logger.info(f"{log_prefix} from session {session_id}: {question}")
    
    current_session = session_id or "default"
    
    # Store location context if it's a short location query
    if any(loc in question.lower() for loc in ["quận", "phường", "đường"]):
        await update_session_context(session_id, question)
        logger.info(f"Stored location context for session {session_id}: {question}")

    # Add reflection check after question processing
    needs_details, reflection_msg = analyze_query(question, lang)
    if needs_details:
        logger.info(f"Query too short, storing context and requesting details: {question}")
        return StreamingResponse(
            iter([reflection_msg]),
            media_type="text/plain"
        )

    # Get chat history and format it
    chat_history = []
    chat_context = ""  # Initialize chat_context
    try:
        messages = message_history.messages
        logger.info(f"Retrieved {len(messages)} messages from history")
        
        # Only get last 3 exchanges (6 messages)
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        for msg in recent_messages:
            chat_history.append(f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}")
            logger.info(f"History message: {msg.type} - {msg.content[:50]}...")
        
        chat_context = "\n".join(chat_history)  # Create chat context string
        logger.info(f"Built chat context with {len(chat_history)} messages")
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")

    async def response_generator():
        try:
            # Add user message to history synchronously
            message_history.add_user_message(question)

            # First, classify the query
            query_type, confidence = await router.classify_query(question, lang)
            logger.info(f"Query classified as: {query_type.value} (confidence: {confidence:.2f})")

            # Get stored context if exists
            stored_context = None 
            if session_id:
                stored_context = await get_session_context(session_id)
                if stored_context:
                    logger.info(f"Retrieved stored context: {stored_context}")

            # Handle different query types
            if query_type == QueryType.GREETING:
                yield llm.prompts[lang]['greeting']
                return
                
            if query_type == QueryType.OUT_OF_SCOPE:
                yield llm.prompts[lang]['no_info']
                return
                
            if query_type == QueryType.FOLLOW_UP:
                if not session_id:
                    yield llm.prompts[lang]['no_memory']
                    return
                    
            if query_type == QueryType.NEEDS_CLARIFICATION:
                if lang == 'vi':
                    yield "Bạn muốn biết thông tin gì về khu vực này? Ví dụ: địa điểm du lịch, ăn uống, mua sắm, di tích lịch sử..."
                else:
                    yield "What would you like to know about this area? For example: tourist spots, food, shopping, historical sites..."
                return

            # For SAIGON and valid FOLLOW_UP queries, continue with RAG process
            log_start = "🔄 Bắt đầu xử lý" if lang == 'vi' else "🔄 Starting RAG process" 
            logger.info(f"{log_start} for session {current_session}")

            # Special handling for follow-up questions
            if query_type == QueryType.FOLLOW_UP:
                if not session_id:
                    if lang == 'vi':
                        yield "Xin lỗi, tôi cần duy trì phiên để hiểu rõ ngữ cảnh câu hỏi của bạn."
                    else:
                        yield "Sorry, I need a session to understand your question context."
                    return

                stored_context = await get_session_context(session_id)
                if stored_context:
                    # Combine stored location with followup question
                    enhanced_question = f"{stored_context} - {question}"
                    logger.info(f"Enhanced question with context: {enhanced_question}")
                    
                    docs = await hyde.enhanced_retrieval(
                        retriever=retriever,
                        question=enhanced_question,
                        lang=lang
                    )
                else:
                    if lang == 'vi':
                        yield "Bạn có thể nói rõ địa điểm bạn muốn tìm hiểu được không?"
                    else:
                        yield "Could you specify which location you're interested in?"
                    return

            else:
                # For non-follow-up questions, update the current topic
                if session_id and question:
                    await update_session_context(session_id, question)
                
                docs = await hyde.enhanced_retrieval(
                    retriever=retriever,
                    question=question,
                    lang=lang,
                    chat_history=chat_history if chat_history else None,
                    stored_context=stored_context
                )

            if not docs:
                logger.info("❌ No relevant documents found in database")
                try:
                    # Initial message about searching web
                    loading_msg = "Không có thông tin trên dữ liệu. Hãy đợi tôi search trên web..."
                    yield loading_msg + "\n\n"
                    await asyncio.sleep(0.5)

                    # Enhance search query for metro-related questions
                    search_query = question
                    if any(term in question.lower() for term in ["metro", "ga", "tàu điện ngầm"]):
                        search_query = f"cách đi đến {question} thông tin chi tiết mới nhất"
                        
                    # Fallback to Tavily Search with enhanced query
                    search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"),    max_results=2)
                    tavily_results = await search.ainvoke(search_query)
                    
                    if tavily_results and isinstance(tavily_results, list):
                        web_context = []
                        sources = []
                        
                        for result in tavily_results:
                            if not isinstance(result, dict):
                                continue
                                
                            content = result.get('content', '')
                            url = result.get('url', '')
                            title = result.get('title', 'Untitled')
                            
                            if content and url:
                                web_context.append(content)
                                sources.append({
                                    'title': title,
                                    'url': url
                                })
                        
                        if web_context:
                            combined_context = "\n\n".join(web_context)
                            logger.info(f"Found alternative information from web search")
                            
                            messages = [
                                SystemMessage(content=f"""Bạn là hướng dẫn viên du lịch Sài Gòn. Hãy trả lời bằng tiếng Việt dựa trên thông tin tìm được:

                                Quy tắc:
                                - Trả lời bằng tiếng Việt có dấu
                                - Giọng điệu thân thiện
                                - Chỉ dùng thông tin từ ngữ cảnh được cung cấp
                                - Không đề cập đến nguồn thông tin trong câu trả lời

                                Ngữ cảnh: {combined_context}"""),
                                HumanMessage(content=question)
                            ]
                            
                            # Stream the response
                            full_response = ""
                            async for chunk in llm.stream_chat(messages):
                                if hasattr(chunk, 'content'):
                                    yield chunk.content
                                    full_response += chunk.content
                            
                            # Add web sources in a more readable format
                            yield "\n\nNguồn tham khảo:\n"
                            for i, source in enumerate(sources, 1):
                                url = source['url']
                                title = source['title'] if source['title'] != 'Untitled' else url.split('//')[-1].split('/')[0]
                                # Format as clickable link with better styling
                                yield f'<div class="source-link">'
                                yield f'<span class="source-number">[{i}]</span> '
                                yield f'<a href="{url}" target="_blank" class="source-url" data-title="{url}">{title}</a>'
                                yield '</div>\n'
                            return

                    # If no results found
                    yield "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
                    return
                            
                except Exception as e:
                    logger.error(f"Error in web search fallback: {str(e)}")
                    yield "Xin lỗi, không tìm thấy thông tin phù hợp."
                    return

            # 2. Build context from documents
            context = []
            metadata = []
            for doc in docs:
                context.append(doc.page_content)
                if hasattr(doc, 'metadata'):
                    metadata.append(doc.metadata)
                    page = doc.metadata.get('page', 'N/A')
                    logger.info(f"Added content from page {page}")

            combined_context = "\n".join(context)
            logger.info(f"📝 Built context length: {len(combined_context)} characters")

            # 3. Generate response using context and history
            logger.info("Generating response with context and history...")
            
            messages = [
                SystemMessage(content=f"""You must ONLY use information from this context and follow these rules strictly:
                
                CONTENT RULES:
                - Only use facts from the provided context
                - DO NOT include any citations or sources in your response
                - DO NOT mention page numbers or references
                - Give direct answers without mentioning where the information comes from
                - You MUST respond in {'Vietnamese' if lang == 'vi' else 'English'} language
                - If language is Vietnamese (vi), use formal Vietnamese with proper diacritics
                - Consider the chat history to maintain conversation context
                
                RECENT CHAT HISTORY:
                {chat_context}
                
                CONTEXT:
                {combined_context}
                
                Current language: {lang}
                Response language: {'Vietnamese' if lang == 'vi' else 'English'}
                """),
                HumanMessage(content=question)
            ]

            # Track response and build full text
            full_response = ""
            async for chunk in llm.stream_chat(messages):
                if chunk.content:
                    # Xử lý từng ký tự thay vì từng từ
                    text = chunk.content
                    for char in text:
                        yield char
                        # Chỉ delay sau dấu cách hoặc dấu câu
                        if char in (' ', '.', '!', '?', ','):
                            await asyncio.sleep(0.05)  # 50ms delay
                            
                    # Thêm xuống dòng nếu kết thúc câu
                    if text.strip().endswith(('.', '!', '?')):
                        yield "\n"
                        await asyncio.sleep(0.2)  # 200ms delay sau mỗi câu
                        
                    # Build full response
                    full_response += text

            # Add assistant's response to history synchronously
            message_history.add_ai_message(full_response)  # Removed await

            # Check if response indicates no information before yielding
            no_info_indicators = [
                "Based on the provided text, there is no mention of",
                "The provided text does not contain information",
                "I don't have information",
                "I don't have that information",
                "Tôi không có thông tin",
                "Văn bản không chứa thông tin",
                "Dựa trên văn bản, không có thông tin"
            ]
            
            # Check for no info response
            if any(indicator in full_response for indicator in no_info_indicators):
                # Clean up response by removing any source citations
                clean_response = full_response.split("Sources:")[0].strip()
                return
                
            # If there is information, continue with normal processing...
            has_info = not any(indicator in full_response for indicator in no_info_indicators)
            
            # Only add citations if response contains actual information
            if has_info and metadata:
                await asyncio.sleep(0.3)
                yield "\n\nSources:\n"
                valid_pages = set()
                for meta in metadata:
                    if 'page' in meta and meta['page']:
                        try:
                            page = int(meta['page'])
                            valid_pages.add(page)
                        except (ValueError, TypeError):
                            continue
                
                for page in sorted(valid_pages):
                    yield f"[Page: {page+1}]\n"
                    await asyncio.sleep(0.2)  # Delay giữa các sources

        except Exception as e:
            logger.error(f"❌ Error in response generation: {str(e)}")
            error_msg = "Xin lỗi, đã có lỗi xảy ra" if lang == 'vi' else "An error occurred"
            yield f"{error_msg}. Please try again."

    return StreamingResponse(
        response_generator(),
        media_type="text/plain", 
        headers={"X-Accel-Buffering": "no"}
    )

# Sửa route weather API
@app.get("/api/weather")
async def get_weather():
    try:
        weather_api_key = os.getenv("WEATHER_API_KEY")
        if not weather_api_key:
            logger.error("Weather API key not found in environment variables")
            return default_weather_response()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                'http://api.weatherapi.com/v1/current.json',
                params={
                    'key': weather_api_key,
                    'q': 'Ho Chi Minh City',
                    'aqi': 'no'
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Weather API error: {response.text}")
                if response.status_code == 403:
                    logger.error("API key invalid or expired")
                return default_weather_response()

            data = response.json()
            return {
                "temperature": data['current']['temp_c'],
                "condition": data['current']['condition']['text'],
                "humidity": data['current']['humidity'],
                "wind_kph": data['current']['wind_kph'],
                "last_updated": data['current']['last_updated']
            }

    except Exception as e:
        logger.error(f"Error fetching weather: {str(e)}")
        return default_weather_response()

def default_weather_response():
    """Return default weather data when API fails"""
    return {
        "temperature": 30,
        "condition": "Partly cloudy",
        "humidity": 70,
        "wind_kph": 15,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "error": "Using default values - API temporarily unavailable"
    }

@app.post("/change-collection")  # Add this route
async def change_collection(request: CollectionRequest):
    global current_collection, db, retriever
    try:
        # Validate collection name
        if request.collection not in ["my_documents", "traveloka"]:
            raise ValueError("Invalid collection name")
            
        current_collection = request.collection
        
        # Reinitialize database connection with new collection
        database = QdrantRAG(qdrant_api, qdrant_url)
        db = database.load_vector_database(collection_name=current_collection)
        
        # Update retriever configuration
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,                # Changed from 3 to 10
                "score_threshold": 0.68  # Increased from 0.3 to be more selective
            }
        )
        
        logger.info(f"✅ Switched to collection: {current_collection}")
        return {"status": "success", "collection": current_collection}
        
    except Exception as e:
        logger.error(f"❌ Error changing collection: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to change collection: {str(e)}"}
        )

if __name__ == "__main__":
    # Initialize components
    try:
        llm = load_gemini()
        logger.info("✅ Successfully loaded Gemini model")
        
        # Initialize Router
        router = SemanticRouter(llm.llm)
        logger.info("✅ Initialized Semantic Router")
        
        # Initialize HyDe
        hyde = HyDe(llm.llm)
        logger.info("✅ Initialized HyDe retrieval enhancement")
        
        database = QdrantRAG(qdrant_api, qdrant_url)
        db = database.load_vector_database(collection_name="my_documents")
        logger.info("✅ Successfully connected to Qdrant database")
        
        # Cập nhật cấu hình retriever
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,                # Changed from 3 to 10
                "score_threshold": 0.68  # Increased from 0.3 to be more selective
            }
        )
        logger.info("✅ Configured retriever with similarity search (k=10, threshold=0.3)")
        
        # Initialize chains
        question_answering_chain = llm.create_stuff_documents_chain()
        history_aware_retriever = llm.create_history_aware_retriever(retriever)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        logger.info("✅ Successfully initialized RAG chain")
        
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {str(e)}")
        raise

