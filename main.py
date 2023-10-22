import os
from typing import List
import json

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse

from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from supabase import create_client, Client

from apikey import apikey
from search_all_stock_tickers import Stock, StockSearchEngine
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import openai

#Include openai api key and supabase api key

supabase: Client = create_client(API_URL, API_KEY)

loader = TextLoader('learn_docs/lesson1.txt')
docs = loader.load()

index = VectorstoreIndexCreator().from_loaders([loader])

app = FastAPI(title = "Youth Investment Learning App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizQuestion(BaseModel):
  question: str = Field(description = "The question to ask the teenager")
  option1: str = Field(description = "The first unique and silly option for the generated question which is not the correct response")
  option2: str = Field(description = "The second unique and tricky option for the generated question which is not the correct response")
  option3: str = Field(description = "The third unique and silly option for the generated question which is not the correct response")
  answer: str = Field(description = "The fourth unique option for the generated question which is the correct response") 
  explanation: str = Field(description = "Explanation of the answer in the context of the given input")

################################################################
@app.get("/")
def main():
  return RedirectResponse(url="/docs")

################################################################

@app.get("/quizQuestion")
def get_quiz_question():
  parser = PydanticOutputParser(pydantic_object = QuizQuestion)
  prompt = PromptTemplate(
    template = "Answer the user query.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
  )

  _input = prompt.format_prompt(query = "Generate a quiz question using the context in the text given here.")
  response = index.query(_input.to_string(), llm = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613"))
  
  return parser.parse(response)

################################################################

#@app.post("/learnBot")
async def learning_content(userId: str):
  # Get user's level from db
  user_level = supabase.table("userData").select("*").eq("id", userId).execute()

  topics = ["money", "saving", "stocks"]
  #if user_level >= 0 and user_level <= 5:
    #Generate easy learning content from level 1 topics
  #elif user_level >= 5 and user_level <=20:
    #Generate content from level 2 topics
  #elif user_level >= 20 and user_level <= 50:
    #Generate final level content


  #Implement chatbot conversational capability

  prompt1 = PromptTemplate(
    template = "Display the following {topics} as user options pulling from the current context",
    input_variables=["topics"],
  )

  prompt2 = PromptTemplate(
    template = "Generate educational content for a teen user on {topic}",
    input_variables=["topic"],
  )

  llm = ChatOpenAI(temperature = 0.5, model = "gpt-3.5-turbo-0613")
  
  #conversation = ConversationChain(
   # prompt = prompt1,
    #llm = llm,
    #verbose = True,
    #memory = ConversationBufferMemory(ai_prefix = "LearnBot")
    #)
  
  #response = conversation.predict(topics = topics) 
  #return response

################################################################

@app.get("/stock/info")
def get_stock_info(ticker: str):
  stock = Stock(ticker)
  stock_info = stock.get_info()
  return jsonable_encoder(stock_info)

################################################################

@app.get("/stock/list_all_tickers")
def list_all_tickers():
  engine = StockSearchEngine()
  tickers = engine.list_all_tickers(page = 1, per_page = 5)
  return tickers

################################################################

@app.get("/stock/multiple_info")
def get_multiple_stock_info(tickers):

  #Check implementation with Krystal
  
  ticker_list = tickers.split(",")
  engine = StockSearchEngine()
  stock_info = engine.get_multiple_stock_info(tickers)
  return stock_info

################################################################

#Implement access to supabase db CRUD operations

#@app.post("/db/insert")
def create_new_user():
  data, count = supabase.table('userData').insert({"id": 20001, "age": 15, "level" : 1, "points" : 1000}).execute()

#@app.get("/db/get_by_id")
def get_row_by_userid():
  response = supabase.table('userData').select('*').eq('id', 20001).execute()
  return response

#@app.get("/db/get_all")
def get_all_rows():
  response = supabase.table('userData').select('*').execute()
  return response

#@app.post("/db/update")
def update_row():
  data, count = supabase.table('userData').update({"age": 16}).eq('id', 20001).execute()

#@app.post("/db/delete")
def delete_row():
  data, count = supabase.table('userData').delete().eq('id', 20001).execute()

################################################################

@app.post("/stock/trend")
def get_stock_trend(ticker: str):
  stock = yf.Ticker(ticker)
  stock_history = stock.history(period = "1y", interval = "1mo")
  stock_history = stock_history.drop(columns = ["Open"])
  stock_history = stock_history.drop(columns = ["Close"])
  stock_history = stock_history.drop(columns = ["Volume"])
  stock_history = stock_history.drop(columns = ["Dividends"])
  stock_history = stock_history.drop(columns = ["Stock Splits"])
  
  try:
      encoded_stock_history = stock_history.reset_index().to_dict('records')
      for data_point in encoded_stock_history:
        data_point["Average"] = (data_point["High"] + data_point["Low"]) / 2
    
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Error encoding stock history data: {e}")

  return jsonable_encoder(encoded_stock_history)


################################################################

#Home- n/a
#About = n/a
#E-learning = chatbot (get first chat message, get topic and display contents for each topic, implement conversational chatbot)
#Quizzes = generate multiple choice quiz based on learning content, quiz difficulty based on user knowledge level
#User registration = insert new user into db
#News = n/a
#Stocks = buy shares, trade shares, display stock trends for 1 stock, display list of all stocks owned by user, display info on all stocks

################################################################

#@app.post("/stock/buy_new")
async def buy_new_stocks(ticker: str, quantity: int, price: int, user_id : str):
  
  stock = Stock(ticker)
  stock_info = stock.get_info()
  stock_price = stock_info["currentPrice"]
  total_cost = stock_price * quantity

  #Assumes that front end verifies there are enough points to buy
  #Update the db with new stock info and update point balance
  data,count = supabase.table('userData').select("*").eq('id', user_id).execute()

  point_balance = data[1][0]['points'] - total_cost
  data, count = supabase.table('userData').update({"points": point_balance}).eq('id', user_id).execute()
  data1, count1 = supabase.table('userData').update({"stocks": {"ticker": ticker, "quantity": quantity, "equity": total_cost}}).eq('id', user_id).execute()

  return data

###################################################################
