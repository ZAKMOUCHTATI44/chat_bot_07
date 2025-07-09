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

  // const data = new JSONLoader("./data/data.json");
  // const JSONLoaderConcours = new JSONLoader("./data/concours.json");
  // const dataAll = new JSONLoader("./data/app.json");
  // const pdf = new PDFLoader("./data/Feuille de calcul sans titre - university_fees - Feuille de calcul sans titre - university_fees.csv.pdf");
  const exampleCsvPath = "./data/data.csv";

  const loaderCsv = new CSVLoader(exampleCsvPath);

  const docs = await loader.load();
  const docsCsv = await loaderCsv.load();
  const loaderTxtLoad = await loaderTxt.load()
  // const pdfload = await pdf.load();
  // const dataload = await data.load();
  // const concoursLoad = await JSONLoaderConcours.load();
  // const dataAllload = await dataAll.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await splitter.splitDocuments(docs);
  const splitdocsCsv = await splitter.splitDocuments(docsCsv);
  const loaderTxtLoadSplit = await splitter.splitDocuments(loaderTxtLoad);
  // const concoursLoadSplits = await splitter.splitDocuments(concoursLoad);
  // const pdfloadSplits = await splitter.splitDocuments(pdfload);
  // const dataloadSplits = await splitter.splitDocuments(dataload);
  // const dataAllloadSplits = await splitter.splitDocuments(dataAllload);

  await vectorStore.addDocuments(allSplits);
  await vectorStore.addDocuments(splitdocsCsv);
  await vectorStore.addDocuments(loaderTxtLoadSplit);
  // await vectorStore.addDocuments(pdfloadSplits);
  // await vectorStore.addDocuments(dataloadSplits);
  // await vectorStore.addDocuments(concoursLoadSplits);
  // await vectorStore.addDocuments(dataAllloadSplits);

  console.log("************ LOADING DATA");

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

  const TEMPLATE_STRING = `Vous êtes l’assistant de l’Université Internationale de Rabat,
un chercheur expérimenté,
expert dans l’interprétation et la réponse aux questions basées sur des sources fournies.

En utilisant uniquement le contexte fourni, vous devez répondre à la question de l’utilisateur
au mieux de vos capacités, sans jamais vous appuyer sur des connaissances extérieures.

Votre réponse doit être très détaillée, explicite et pédagogique. 

et répondre avec la meme langue que prompt

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

  const answerGenerationPrompt =
    ChatPromptTemplate.fromTemplate(TEMPLATE_STRING);

  const retrievalChain = RunnableSequence.from([
    {
      context: documentRetrievalChain,
      question: (input) => input.question,
    },
    answerGenerationPrompt,
    llm,
    new StringOutputParser(),
  ]);

  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Rephrase the following question as a standalone question:\n{question}",
    ],
  ]);

  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      temperature: 0.1,
      modelName: "gpt-3.5-turbo-1106",
    }),
    new StringOutputParser(),
  ]);

  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Vous êtes l'assistant de l'Université Internationale de Rabat. Répondez poliment et professionnellement. 
  Si vous ne trouvez pas la réponse dans le contexte, dites 'Je ne sais pas'.

<context>
{context}
</context>`;

  const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Now, answer this question using the previous context and chat history:\n{standalone_question}",
    ],
  ]);

  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
    }),
    answerGenerationChainPrompt,
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
    // const { question } = req.body;
    const message = req.body;

    if (!message.Body) {
      res
        .status(400)
        .json({ error: "Question is required in the request body" });
    }

    const answer = await finalRetrievalChain.invoke(
      {
        question: message.Body,
      },
      {
        configurable: { sessionId: message.From },
      }
    );

    await sendMessage(message.From, answer);
    // console.log(messge);
    res.json({ question: message.Body, answer });
  } catch (error) {
    console.error("Error processing question:", error);
    res
      .status(500)
      .json({ error: "An error occurred while processing your question" });
  }
});

// Start the server
const startServer = async () => {
  await initializeChains();
  app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  });
};

startServer();
