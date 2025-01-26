import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
import logging
from typing import Optional  # Add this import

logger = logging.getLogger(__name__)

def load_config_mongo_db():
    load_dotenv()
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGODB_URI = f"mongodb+srv://trinhquocbao27:{MONGO_PASSWORD}@sessionhistory.tcnf8.mongodb.net/?retryWrites=true&w=majority&appName=SessionHistory"
    DB_NAME="SessionHistory"
    collection_name="chat_history"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
    return MONGO_PASSWORD, MONGODB_URI, DB_NAME, collection_name, ATLAS_VECTOR_SEARCH_INDEX_NAME

class SessionContext:
    def __init__(self, messages=None, current_topic=None):
        self.messages = messages or []
        self.current_topic = current_topic
    
def get_session_history(session_id: str):
    """Get chat history for a session with automatic cleanup of old messages."""
    MONGO_PASSWORD, MONGODB_URI, DB_NAME, collection_name, _ = load_config_mongo_db()
    
    history = MongoDBChatMessageHistory(
        connection_string=MONGODB_URI,
        session_id=session_id,
        database_name=DB_NAME,
        collection_name=collection_name,
    )
    
    try:
        # Initialize the collection if it doesn't exist
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            db[collection_name].create_index("session_id")
            db[collection_name].create_index("created_at")
        
        # Ensure session exists
        if session_id:
            db[collection_name].update_one(
                {"session_id": session_id},
                {"$setOnInsert": {"created_at": datetime.now()}},
                upsert=True
            )
        
        # Store current topic for context persistence
        context_collection = db["session_context"]
        if session_id:
            context = context_collection.find_one({"session_id": session_id})
            if context:
                history.current_topic = context.get("current_topic")
        
        client.close()
        
        # Cleanup old messages
        cleanup_old_messages(history)
        
    except Exception as e:
        logger.error(f"Error initializing MongoDB collection: {str(e)}")
    
    return history

def cleanup_old_messages(history: MongoDBChatMessageHistory):
    """Keep only the last 3 messages in the chat history."""
    if not history.session_id:
        return
        
    try:
        MONGO_PASSWORD, MONGODB_URI, DB_NAME, collection_name, _ = load_config_mongo_db()
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[collection_name]
        
        # Get all messages for this session ordered by timestamp
        cursor = collection.find(
            {"session_id": history.session_id}
        ).sort("created_at", -1)
        
        messages = list(cursor)
        
        if len(messages) > 3:
            # Keep only the last 3 messages
            keep_ids = [msg["_id"] for msg in messages[:3]]
            collection.delete_many({
                "session_id": history.session_id,
                "_id": {"$nin": keep_ids}
            })
        
        client.close()
        
    except Exception as e:
        logger.error(f"Error cleaning up messages: {str(e)}")

def clear_chat_history(session_id: str = None):
    """Clear chat history for a specific session or all sessions."""
    MONGO_PASSWORD, MONGODB_URI, DB_NAME, collection_name, _ = load_config_mongo_db()
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[collection_name]
    
    if session_id:
        # Delete history for specific session
        result = collection.delete_many({"session_id": session_id})
    else:
        # Delete all history
        result = collection.delete_many({})
    
    client.close()
    return result.deleted_count

def cleanup_old_sessions():
    """Clean up sessions older than 24 hours."""
    MONGO_PASSWORD, MONGODB_URI, DB_NAME, collection_name, _ = load_config_mongo_db()
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[collection_name]
    
    # Delete messages older than 24 hours
    yesterday = datetime.now() - timedelta(days=1)
    result = collection.delete_many({"created_at": {"$lt": yesterday}})
    
    client.close()
    return result.deleted_count

async def get_session_context(session_id: str) -> Optional[str]:
    """Get stored context for a session."""
    try:
        MONGO_PASSWORD, MONGODB_URI, DB_NAME, _, _ = load_config_mongo_db()
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        context = db.session_context.find_one({"session_id": session_id})
        client.close()
        
        if context:
            return context.get("current_topic")
    except Exception as e:
        logger.error(f"Error getting session context: {str(e)}")
    return None

async def update_session_context(session_id: str, context: str):
    """Update session context, especially for location queries."""
    if not session_id:
        return
        
    try:
        MONGO_PASSWORD, MONGODB_URI, DB_NAME, _, _ = load_config_mongo_db()
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        db.session_context.update_one(
            {"session_id": session_id},
            {"$set": {
                "current_topic": context,
                "updated_at": datetime.now()
            }},
            upsert=True
        )
        client.close()
    except Exception as e:
        logger.error(f"Error updating session context: {str(e)}")


