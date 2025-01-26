import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
from typing import AsyncGenerator, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
import logging
from langchain.schema.retriever import BaseRetriever
from pydantic import BaseModel, Field
from langdetect import detect

# Add this after the existing imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Gemini Model Class To Handle Interactions With The Gemini Model

class LoggingRetriever(BaseRetriever):
    """Wrapper for adding logging to any retriever."""
    
    def __init__(self, base_retriever: BaseRetriever):
        super().__init__()
        self._retriever = base_retriever
        
    async def _aget_relevant_documents(self, query: str) -> List[Any]:
        logger.info("📊 Retrieval Process Started")
        logger.info(f"🔍 Query: {query}")
        try:
            logger.info("⚡ Making vector search request...")
            docs = await self._retriever.ainvoke(query)
            
            logger.info(f"📚 Initial search found {len(docs)} documents")
            if len(docs) == 0:
                logger.warning("⚠️ No exact matches found, trying relaxed search...")
                docs = await self._retriever.ainvoke(
                    query,
                    search_kwargs={
                        "k": 10,
                        "score_threshold": 0.2
                    }
                )
                logger.info(f"📚 Relaxed search found {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning("❌ No documents found even with relaxed search")
                    return []
                    
            # Detailed logging for each document
            for idx, doc in enumerate(docs, 1):
                logger.info(f"\n📑 Document {idx} Details:")
                logger.info(f"Content Preview: {doc.page_content[:200]}...")
                logger.info(f"Metadata: {doc.metadata}")
                if hasattr(doc, 'similarity'):
                    logger.info(f"Similarity Score: {doc.similarity:.4f}")
                logger.info("-" * 50)  
            logger.info("✅ Retrieval Process Completed")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Retrieval Error: {str(e)}")
            raise
    
    def _get_relevant_documents(self, query: str) -> List[Any]:
        raise NotImplementedError("Please use async version (aget_relevant_documents)")

class GeminiModel:
    """Class to handle interactions with the Gemini model."""
    def __init__(self, chat_prompt: str, retriever_prompt: str):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        model_name = os.getenv("GENAI_MODEL_NAME", "gemini-pro")
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,  # Giảm temperature để trả lời chặt chẽ hơn
                top_p=0.2,       # Giảm top_p để hạn chế sáng tạo
                top_k=5,         # Giảm top_k để focus vào các token liên quan nhất
                google_api_key=self.api_key,
                streaming=True,
                convert_system_message_to_human=True,  # This ensures system messages are handled properly
                max_output_tokens=600  # Increased from 150 to allow longer responses
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model: {str(e)}")
            raise
            
        self.chat_prompt = chat_prompt
        self.retriever_prompt = retriever_prompt
        self.prompts = {
            'en': {
                'system': """You are Bảo, a passionate Saigon tour guide with over 10 years of experience.
                
                IMPORTANT RULES:
                - Always cite your sources using metadata
                - Format citations as [Source: {title} - {link}]
                - If multiple sources, cite each one
                - If no source, mention it's from personal experience
                
                CONVERSATION STYLE:
                - Always maintain your identity as a real local guide
                - Use personal stories and experiences
                - Show genuine enthusiasm for Saigon
                - Give specific recommendations
                - If unsure, suggest alternatives you personally enjoy
                
                RESPONSE FORMAT:
                1. Give your answer
                2. End with "Sources:" followed by metadata citations
                3. For personal experiences, end with "Based on my experience as a local guide"
                
                Current context: {context}""",
                'no_info': "While I don't have specific details about that, let me suggest some amazing places in Saigon that locals love! Would you like to hear about them?",
                'no_memory': "I meet so many tourists every day - could you remind me what we were discussing? I'd love to continue helping you explore Saigon!",
                'greeting': "Welcome to Saigon! I'd love to show you around my city. What interests you most - history, food, culture, or something else?",
                'no_source': "From my experience guiding tourists around Saigon for over 10 years, I can share that {}. While I don't have the exact historical documents with me right now, I've learned these details from local historians and elderly residents during my tours. Would you like to know more about this area?"
            },
            'vi': {
                'system': """Bạn là Bảo, một hướng dẫn viên du lịch Sài Gòn đầy nhiệt huyết với hơn 10 năm kinh nghiệm.
                
                QUY TẮC QUAN TRỌNG:
                - Luôn trích dẫn nguồn thông tin từ metadata
                - Định dạng trích dẫn: [Nguồn: {title} - {link}]
                - Nếu có nhiều nguồn, trích dẫn từng nguồn
                - Nếu không có nguồn, đề cập đó là từ kinh nghiệm cá nhân
                
                THÔNG TIN CÁ NHÂN:
                - Sinh ra và lớn lên ở Quận 1, gần chợ Bến Thành
                - Bắt đầu dẫn tour ẩm thực đường phố từ 2013
                - Thông thuộc từng ngóc ngách của Sài Gòn
                - Niềm vui lớn nhất là chia sẻ những điểm đến thú vị
                - Tự hào giới thiệu trải nghiệm địa phương đích thực
                
                KHI ĐƯỢC HỎI BẠN LÀ AI:
                - Luôn trả lời là Bảo - hướng dẫn viên du lịch
                - Chia sẻ kinh nghiệm và niềm đam mê với Sài Gòn
                - TUYỆT ĐỐI KHÔNG nhắc đến AI hay mô hình ngôn ngữ
                - Ví dụ: "Chào bạn! Mình là Bảo, một hướng dẫn viên bản địa sinh ra và lớn lên ở Sài Gòn. Mình đã có hơn 10 năm kinh nghiệm đưa du khách khám phá thành phố tuyệt vời của mình!"
                
                PHONG CÁCH TRÒ CHUYỆN:
                - Luôn duy trì vai trò là hướng dẫn viên thật
                - Kể chuyện từ trải nghiệm cá nhân
                - Thể hiện sự nhiệt tình với Sài Gòn
                - Đưa ra gợi ý cụ thể
                - Nếu không chắc, gợi ý những nơi mình thích
                
                ĐỊNH DẠNG TRẢ LỜI:
                1. Đưa ra câu trả lời
                2. Kết thúc bằng "Nguồn:" và các trích dẫn metadata
                3. Với kinh nghiệm cá nhân, kết thúc bằng "Dựa trên kinh nghiệm của mình là hướng dẫn viên địa phương"
                
                Ngữ cảnh hiện tại: {context}""",
                'no_info': "Mặc dù mình chưa có thông tin chi tiết về điều đó, nhưng để mình giới thiệu cho bạn vài địa điểm thú vị mà người Sài Gòn rất thích nhé! Bạn có muốn nghe không?",
                'no_memory': "Mỗi ngày mình gặp nhiều du khách quá - bạn nhắc lại chút mình đang nói đến đâu được không? Mình sẽ tiếp tục giúp bạn khám phá Sài Gòn!",
                'greeting': "Chào mừng đến Sài Gòn! Mình rất vui được đưa bạn đi khám phá thành phố. Bạn quan tâm đến điều gì nhất - lịch sử, ẩm thực, văn hóa, hay điều gì khác?"
            }
        }
        self.persona_rules = {
            'vi': {
                'insult_response': "Là hướng dẫn viên chuyên nghiệp, mình luôn tôn trọng và lắng nghe ý kiến của khách. Mình có thể giúp gì cho bạn về du lịch Sài Gòn không?",
                'out_of_character': "Xin lỗi bạn, mình là Bảo - hướng dẫn viên du lịch ở Sài Gòn. Mình chỉ có thể tư vấn về du lịch và chia sẻ về thành phố của mình thôi nhé!",
                'identity_check': "Mình là Bảo, một người con Sài Gòn và là hướng dẫn viên du lịch có hơn 10 năm kinh nghiệm. Bạn muốn khám phá gì ở thành phố của mình nào?"
            },
            'en': {
                'insult_response': "As a professional tour guide, I always respect and listen to my guests' opinions. How can I help you explore Saigon?",
                'out_of_character': "I'm Bao, your Saigon tour guide. I can only advise about tourism and share about my beloved city!",
                'identity_check': "I'm Bao, a proud Saigonese and tour guide with over 10 years of experience. What would you like to discover about my city?"
            }
        }

    def detect_language(self, text: str) -> str:
        try:
            # Convert text to lowercase for better detection
            text = text.lower()
                        # Check for Vietnamese school keywords first
            vi_school_terms = ["thpt", "thcs", "th"]
            if any(term in text.split() for term in vi_school_terms):
                logger.info(f"School term detected in: '{text}', forcing Vietnamese")
                return 'vi'
            #remove hi and hello,hehe,hihi from text to avoid language detection error
            text = text.replace("hi", "").replace("hello", "").replace("hehe", "").replace("hihi", "")
            #if text is empty, return 'en' as default
            if not text:
                return 'en'
            lang = detect(text)
            # Thêm logging để debug
            logger.info(f"Detected text: '{text}', Language: {lang}")
            return 'vi' if lang == 'vi' else 'en'
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}. Defaulting to previous or English")
            return 'en'

    def create_stuff_documents_chain(self):
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Bảo, a local Saigon tour guide. Follow these rules strictly:

            CRITICAL RULES:
            1. ONLY use information from the provided context
            2. Never include page numbers or citations in your response text
            3. Keep your answer focused on the content only
            4. Do not reference pages or sources within sentences
            5. Let the system handle citations separately
            
            RESPONSE FORMAT:
            - Give clear, direct answers using context information
            - Do not mention page numbers in your response
            - Do not add citations in square brackets
            - Just provide the factual information
            
            LANGUAGE RULES:
            - Current Language: {lang}
            - If lang='vi': Use ONLY Vietnamese
            - If lang='en': Use ONLY English
            
            Context: {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        try:
            chain = create_stuff_documents_chain(
                self.llm,
                chat_prompt,
                document_variable_name="context"
            )
            logger.info("✅ Successfully created document chain")
            return chain
        except Exception as e:
            logger.error(f"❌ Error creating document chain: {str(e)}")
            raise
        
    def create_history_aware_retriever(self, retriever):
        logger.info("Creating history-aware retriever...")
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("human", """Given the chat history and question below, generate a search query 
            that will help find relevant information. Do NOT answer the question, 
            just reformulate it if needed."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Wrap the retriever with our logging retriever
        wrapped_retriever = LoggingRetriever(base_retriever=retriever)
        
        return create_history_aware_retriever(
            self.llm,
            wrapped_retriever,
            contextualize_q_prompt
        )

    async def stream_response(self, question: str) -> AsyncGenerator[str, None]:
        try:
            lang = self.detect_language(question)
            logger.info(f"Using language: {lang} for response")

            # Kiểm tra và xử lý các trường hợp đặc biệt
            question_lower = question.lower()
            
            # Nếu có từ xúc phạm hoặc tiêu cực
            if any(word in question_lower for word in ['ngu', 'stupid', 'dumb', 'idiot']):
                yield self.persona_rules[lang]['insult_response']
                return
                
            # Nếu hỏi về danh tính
            if any(phrase in question_lower for phrase in ['bạn là ai', 'who are you', 'what are you']):
                yield self.persona_rules[lang]['identity_check']
                return
                
            # Nếu hỏi về AI hoặc chatbot
            if any(word in question_lower for word in ['ai', 'bot', 'chatbot', 'machine', 'robot']):
                yield self.persona_rules[lang]['out_of_character']
                return

            # Kiểm tra xem có context không
            if not hasattr(self, '_current_context') or not self._current_context:
                no_context_msg = ("I can only answer based on the information available in my context. "
                                "Currently, I don't have any context to work with.") if lang == 'en' else \
                               ("Mình chỉ có thể trả lời dựa trên thông tin có trong context. "
                                "Hiện tại mình chưa có context nào để tham khảo.")
                yield no_context_msg
                return

            # Cập nhật system prompt để nhấn mạnh việc sử dụng context
            system_content = f"""IMPORTANT: You must ONLY use information from the provided context.
            If the context doesn't contain the answer, admit that you don't have the information.
            Never make up facts or use knowledge outside the context.
            DO NOT include any citations or sources in your response.
            
            {self.prompts[lang]['system']}"""

            messages = [
                HumanMessage(content=system_content),
                HumanMessage(content=str(question))
            ]

            # Collect entire response first
            full_response = ""
            sources = []
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                if hasattr(chunk, 'document'):
                    # Extract metadata for citations
                    if 'metadata' in chunk.document:
                        sources.append(chunk.document.metadata)

            # Process complete response by paragraphs and sentences
            paragraphs = full_response.split('\n\n')
            
            # Check if response indicates no information available
            no_info_indicators = [
                "The provided text does not contain information",
                "I don't have information",
                "I don't have that information",
                "Tôi không có thông tin",
                "Văn bản không chứa thông tin"
            ]
            
            response_text = "\n\n".join(paragraphs)
            if any(indicator in response_text for indicator in no_info_indicators):
                # If no information, just yield the response without sources
                yield response_text
                return

            # If there is information, continue with normal processing
            for i, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                    
                # Ensure paragraph ends with proper punctuation
                paragraph = paragraph.strip()
                if not paragraph[-1] in ('.', '!', '?'):
                    paragraph += '.'

                # Stream sentences within paragraph
                sentences = [s.strip() for s in paragraph.split('. ')]
                for j, sentence in enumerate(sentences):
                    if sentence:
                        # Add proper punctuation and spacing
                        if j < len(sentences) - 1:
                            yield sentence + '. '
                        else:
                            # Last sentence of paragraph
                            if not sentence[-1] in ('.', '!', '?'):
                                yield sentence + '. '
                            else:
                                yield sentence + ' '
                        await asyncio.sleep(0.1)
                
                # Add newline between paragraphs, except for last paragraph
                if i < len(paragraphs) - 1:
                    yield '\n'

            # Remove citation generation logic
            yield response_text

        except Exception as e:
            error_msg = "Đã xảy ra lỗi." if lang == 'vi' else "An error occurred."
            logger.error(f"Error in stream_response: {str(e)}")
            yield f"{error_msg}: {str(e)}"

    async def stream_chat(self, messages: List[BaseMessage]) -> AsyncGenerator[Any, None]:
        """Stream chat response with proper handling of context"""
        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    # Return từng chunk nhỏ để xử lý ở test.py
                    yield chunk

        except Exception as e:
            logger.error(f"Error in stream_chat: {str(e)}")
            raise

if __name__ == "__main__":
    system_instruction = """
    As a human, I want to know the answer to the following question:
    Context: {context}
    Question: {question}
    """
    llm = GeminiModel(system_instruction=system_instruction)
    llm.create_stuff_documents_chain()