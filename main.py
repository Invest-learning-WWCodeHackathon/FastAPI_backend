import os
from typing import List
import json

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse

from langchain.chains import LLMChain, ConversationChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from supabase import create_client, Client

from apikey import apikey, supabasekey
from search_all_stock_tickers import Stock, StockSearchEngine
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import openai

os.environ['OPENAI_API_KEY'] = apikey

API_URL = 'https://otelrzfgawehviamvzox.supabase.co'
API_KEY = supabasekey

supabase: Client = create_client(API_URL, API_KEY)

loader1 = TextLoader('learn_docs/level1.txt')
#docs1 = loader.load()
index1 = VectorstoreIndexCreator().from_loaders([loader1])

loader2 = TextLoader('learn_docs/level2.txt')
index2 = VectorstoreIndexCreator().from_loaders([loader2])

loader3 = TextLoader('learn_docs/level3.txt')
index3 = VectorstoreIndexCreator().from_loaders([loader3])

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
def get_quiz_question(user_id: str):

  data, count = supabase.table('userData').select('*').eq("user_id", user_id).execute()

  level = data[1][0]["level"]
    
  parser = PydanticOutputParser(pydantic_object = QuizQuestion)
  prompt = PromptTemplate(
    template = "Answer the user query.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
  )

  _input = prompt.format_prompt(query = "Generate a question using the context in the text given here.")
  
  if level == 1:
    response = index1.query(_input.to_string(), llm = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613"))
  elif level == 2:
    response = index2.query(_input.to_string(), llm = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613"))
  elif level == 3:
    response = index3.query(_input.to_string(), llm = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613"))

  return parser.parse(response)

################################################################

@app.post("/learnBot")
async def learning_content(userId: str, userInput: str):
  # Get user's level from db
  data, count = supabase.table("userData").select("*").eq("user_id", userId).execute()

  user_level = data[1][0]["level"]

  loader = TextLoader('learn_docs/level1.txt')
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
  docs = text_splitter.split_documents(documents)

  embeddings = OpenAIEmbeddings()
  db = DocArrayInMemorySearch.from_documents(docs, embeddings)
  retriever = db.as_retriever(search_type = "similarity", search_kwargs={"k":4})
  
  # Create chatbot for level 1 users

  if user_level == 1:

    chat_template = ChatPromptTemplate.from_messages([
      ("system", "You are a teaching assistant providing comprehensive lessons for teenagers. Provide lessons geared towards 17 year olds in simple language and abundant examples. First introduce yourself as LearnBot and then give the user a list of topics from:\n1. What Is a Stock? An Introduction for Teens\n2.Why Do Companies Issue Stocks?\n3.How to Buy Your First Stock as a Teenager\n4.The Role of Stock Exchanges: NYSE, NASDAQ, and Others\n5.What Happens at a Stock Market Opening and Closing?.\n Based on the user's input, generate the next message. Ask the user questions to make them think and learn the concepts in greater detail. Use the sample conversation below to guide you:\n LearnBot: Hello! I'm LearnBot, your helpful teaching assistant. Choose a topic you'd like to learn about today. 1. What is a stock? 2. Why do companies issue stocks? 3. What is the role of the stock exchange?\nUser: Tell me more about what stocks are \n LearnBot: Sure. Let us begin with an analogy *end of example* and continue the conversation in the same manner"),
      ("human", "What is the topic of the lesson?"),
      ("ai", "What are stocks"),
      ("human", "{user_input}"),
    ])
    
    llm = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613")
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    
    conversation = LLMChain(llm = llm, prompt = chat_template, verbose = True, memory = memory)
    response = conversation({"user_input": userInput})    
    
    return response

  
################################################################

@app.get("/stock/info")
def get_stock_info(ticker: str):
  stock = Stock(ticker)
  stock_info = stock.get_info()
  return jsonable_encoder(stock_info)

################################################################

@app.get("/stock/price")
def get_stock_price(ticker: str):
  stock = yf.Ticker(ticker)
  stock_price = stock.info["currentPrice"]
  return jsonable_encoder(stock_price)
  
################################################################

@app.get("/stock/list_all_tickers")
def list_all_tickers(pages: int, per_page: int):
  engine = StockSearchEngine()
  tickers = engine.list_all_tickers(page = pages, per_page = per_page)
  return tickers

################################################################

#@app.get("/stock/multiple_info")
def get_multiple_stock_info(tickers):

  #Check implementation with Krystal
  
  ticker_list = tickers.split(",")
  engine = StockSearchEngine()
  stock_info = engine.get_multiple_stock_info(tickers)
  return stock_info

################################################################

#Implement access to supabase db CRUD operations

@app.post("/db/insert")
def create_new_user(user_id: str):

  uData, uCount = supabase.table('userData').select("*", count = 'exact').eq("user_id", user_id).execute() 

  if uCount[1] == 0 or uCount[1] == None:
    #User is new
    data, count = supabase.table('userData').insert({"user_id": user_id, "level" : 1, "points" : 1000}).execute()
    return jsonable_encoder({"New User": True})
  else:
    return jsonable_encoder({"New User": False})
    
################################################################

@app.get("/db/get_by_id")
def get_row_by_userid(user_id: str):
  response = supabase.table('userData').select('*').eq('user_id', user_id).execute()
  return jsonable_encoder(response)

################################################################

#@app.get("/db/get_all")
def get_all_rows():
  response = supabase.table('userData').select('*').execute()
  return response

################################################################

#@app.post("/db/update")
def update_row():
  data, count = supabase.table('userData').update({"age": 16}).eq('id', 20001).execute()




#@app.post("/db/delete")
def delete_row():
  data, count = supabase.table('userData').delete().eq('id', 20001).execute()

################################################################

@app.get("/stock/trend")
def get_stock_trend(ticker: str):
  stock = yf.Ticker(ticker)
  stock_info = stock.info
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

      encoded_stock_history.append({"Name": stock_info["longName"], "CurrentPrice" : stock_info["currentPrice"]})
    
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

@app.get("/stock/buy_stocks")
async def buy_stocks(ticker: str, quantity: float, buying_price: float, current_points: float, user_id : str):

  userdata, usercount = supabase.table('userData').update({"points" : current_points}).eq('user_id', user_id).execute()
  stockdata, stockcount = supabase.table('stockData').select('*', count = 'exact').eq('user_id', user_id).eq('ticker', ticker).execute()

  #print(stockdata)
  #print(stockcount)
  
  if (stockcount[1] == None) or (stockcount[1] == 0):
    print("new stock for this user")
    data, count = supabase.table('stockData').insert({"user_id": user_id, "ticker": ticker, "quantity": quantity, "equity": (quantity * buying_price)}).execute()
    return True
  
  elif stockcount[1] > 0:
    #Stocks from this company have been bought.. update values instead
    print("stock already exists, updating counts")
    current_quantity = stockdata[1][0]['quantity']
    updated_quantity = current_quantity + quantity
    equity_gained = buying_price*quantity
    updated_equity = stockdata[1][0]['equity'] + equity_gained
    data, count = supabase.table('stockData').update({"quantity" : updated_quantity, "equity": updated_equity, "buying_price": buying_price}).eq('user_id', user_id).eq('ticker', ticker).execute()
    return True

  return False

###################################################################

@app.get("/stock/sell_stocks")
async def sell_stocks(ticker: str, quantity: float, selling_price: float, current_points: int, user_id: str):
  userdata, usercount = supabase.table('userData').update({"points" : current_points}).eq('user_id', user_id).execute()
  stockdata, stockcount = supabase.table('stockData').select('*', count = 'exact').eq('user_id', user_id).eq('ticker', ticker).execute()
  
  if (stockcount[1] == None) or (stockcount[1] == 0):
    print("no stocks to sell")
    return False
  
  elif stockcount[1] > 0:
    #Stocks from this company have been sold.. update values instead
    print("stock already exists, updating counts")
    current_quantity = stockdata[1][0]['quantity']
    updated_quantity = current_quantity - quantity
    equity_lost = selling_price*quantity
    updated_equity = stockdata[1][0]['equity'] - equity_lost
    data, count = supabase.table('stockData').update({"quantity" : updated_quantity, "equity": updated_equity}).eq('user_id', user_id).eq('ticker', ticker).execute()
    return True
    
  return False

###################################################################

@app.get("/stock/get_user_stock_portfolio")
async def get_user_stock_portfolio(user_id: str):

  data, count = supabase.table('stockData').select('*').eq('user_id', user_id).execute()
  return jsonable_encoder(data)

###################################################################

@app.get("/update_points")
async def update_points(user_id: str, current_points: int, was_correct: bool):

  data, count = supabase.table('userData').select('*').eq('user_id', user_id).execute()
  
  answers = int(data[1][0].get("correct_answers", 0))
  
  if(was_correct):
    answers = answers+1

  if answers > 10:
    new_level = data[1][0]["level"] + 1
  else:
    new_level = data[1][0]["level"]
  
  userdata, usercount = supabase.table('userData').update({"level": new_level, "points" : current_points, "correct_answers" : answers}).eq('user_id', user_id).execute()
  return ("User points updated")

###################################################################

