import "dotenv/config";
import { vectorStore } from "./src/vector";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { llm } from "./src/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import express, { Request, Response } from "express";
import bodyParser from "body-parser";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { sendMessage } from "./src/message";

const app = express();
const port = process.env.PORT || 7001;

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(bodyParser.json());

// Initialize the vector store and chains
let finalRetrievalChain: any;

const initializeChains = async () => {
  const loader = new JSONLoader("./data/faq.json");
  const loaderTxt = new TextLoader("./data/general.txt");
  const catalogueTxt = new TextLoader("./data/Catalogue UIR .txt");
  const loaderCsv = new CSVLoader("./data/data.csv");

  const docs = await loader.load();
  const docsCsv = await loaderCsv.load();
  const loaderTxtLoad = await loaderTxt.load();
  const catalogueTxtLoad = await catalogueTxt.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await splitter.splitDocuments(docs);
  const splitdocsCsv = await splitter.splitDocuments(docsCsv);
  const loaderTxtLoadSplit = await splitter.splitDocuments(loaderTxtLoad);
  const loaderCatalogueTxt = await splitter.splitDocuments(catalogueTxtLoad);

  await vectorStore.addDocuments(allSplits);
  await vectorStore.addDocuments(splitdocsCsv);
  await vectorStore.addDocuments(loaderTxtLoadSplit);
  await vectorStore.addDocuments(loaderCatalogueTxt);

  console.log("************ LOADING DATA COMPLETE");

  const retriever = vectorStore.asRetriever();

  const convertDocsToString = (documents: Document[]): string => {
    return documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
  };

  const documentRetrievalChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  // Simplified templates
  const REPHRASE_QUESTION_TEMPLATE = `Please rephrase the following question to be standalone, keeping it simple and clear. If you're unsure, just return the original question:
  
  Chat history (if relevant):
  {history}
  
  Follow-up question:
  {question}
  
  Standalone question:`;

  const ANSWER_TEMPLATE = `You are the assistant for International University of Rabat. Respond politely and professionally in the same language as the question.
  
  If greeted, respond with: "Hello! I'm the virtual assistant of International University of Rabat. How can I help you today?"
  
  If the answer isn't in the context, simply say: "I don't have that information. Please contact the university directly for more details."
  
  Context:
  {context}
  
  Question:
  {standalone_question}
  
  Answer:`;

  // Simplified chains
  const rephraseQuestionChain = RunnableSequence.from([
    ChatPromptTemplate.fromMessages([
      ["system", REPHRASE_QUESTION_TEMPLATE],
      new MessagesPlaceholder("history"),
      ["human", "{question}"],
    ]),
    new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      temperature: 0.1,
      modelName: "gpt-3.5-turbo-1106",
    }),
    new StringOutputParser(),
  ]);

  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
    }),
    ChatPromptTemplate.fromMessages([
      ["system", ANSWER_TEMPLATE],
      new MessagesPlaceholder("history"),
      ["human", "{standalone_question}"],
    ]),
    llm,
    new StringOutputParser(),
  ]);

  const messageHistory = new ChatMessageHistory();

  finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: (_sessionId) => messageHistory,
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });
};

// Express route to handle questions
app.post("/uir-chat-bot", async (req: Request, res: Response) => {
  try {
    const message = req.body;

    if (!message.Body) {
      res.status(400).json({ error: "Question is required" });
    }

    const answer = await finalRetrievalChain.invoke(
      {
        question: message.Body,
      },
      {
        configurable: { sessionId: `${message.From}-${Date.now()}` },
      }
    );

    await sendMessage(message.From, answer);
    res.json({ question: message.Body, answer });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "An error occurred" });
  }
});

// Start the server
const startServer = async () => {
  await initializeChains();
  app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
  });
};

startServer();
