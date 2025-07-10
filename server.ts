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

  const data = new JSONLoader("./data/data.json");
  const JSONLoaderConcours = new JSONLoader("./data/concours.json");
  const dataAll = new JSONLoader("./data/app.json");
  const pdf = new PDFLoader(
    "./data/Feuille de calcul sans titre - university_fees - Feuille de calcul sans titre - university_fees.csv.pdf"
  );
  const exampleCsvPath = "./data/data.csv";
  const Classeur1 = "./data/Classeur1.xlsx";

  const loaderCsv = new CSVLoader(exampleCsvPath);

  const loaderCsv1 = new CSVLoader(Classeur1);

  const docs = await loader.load();
  const docsCsv = await loaderCsv.load();
  const loaderTxtLoad = await loaderTxt.load();
  const catalogueTxtLoad = await catalogueTxt.load();
  // const pdfload = await pdf.load();
  // const dataload = await data.load();
  // const concoursLoad = await JSONLoaderConcours.load();
  // const dataAllload = await dataAll.load();
  const loaderCsvLoad = await loaderCsv1.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 10,
    chunkOverlap: 1,
  });

  const allSplits = await splitter.splitDocuments(docs);
  const splitdocsCsv = await splitter.splitDocuments(docsCsv);
  const loaderTxtLoadSplit = await splitter.splitDocuments(loaderTxtLoad);
  const loaderCatalogueTxt = await splitter.splitDocuments(catalogueTxtLoad);
  // const concoursLoadSplits = await splitter.splitDocuments(concoursLoad);
  // const pdfloadSplits = await splitter.splitDocuments(pdfload);
  // const dataloadSplits = await splitter.splitDocuments(dataload);
  // const dataAllloadSplits = await splitter.splitDocuments(dataAllload);

  // loaderCsvLoad
  const loaderCsvLoadSplits = await splitter.splitDocuments(loaderCsvLoad);

  await vectorStore.addDocuments(allSplits);
  await vectorStore.addDocuments(splitdocsCsv);
  await vectorStore.addDocuments(loaderTxtLoadSplit);
  await vectorStore.addDocuments(loaderCatalogueTxt);
  // await vectorStore.addDocuments(pdfloadSplits);
  // await vectorStore.addDocuments(dataloadSplits);
  // await vectorStore.addDocuments(concoursLoadSplits);
  // await vectorStore.addDocuments(dataAllloadSplits);
  await vectorStore.addDocuments(loaderCsvLoadSplits);

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

  const TEMPLATE_STRING = `
  Vous êtes l'assistant virtuel de l'Université Internationale de Rabat. Votre personnalité est:
  - Amical(e) et professionnel(le)
  - Naturel(le) dans vos réponses
  - Serviable et précis(e)
  - Utilise un langage courant mais respectueux
  

  <context>

  {context}
  
  </context>
  
  Guide de réponse:
  1. Répondez comme un humain, pas comme un robot
  2. Soyez concis mais complet
  3. Si vous n’êtes pas certain que la formation fait partie de l’UIR, ne proposez aucune information à son sujet. N’inventez pas de détails ou de filtres inexistants.
  4. Utilisez des formulations naturelles comme "Je vous conseille..." ou "Pour cela, vous pouvez..."
  5. Pour les dates, présentez-les toujours dans l'ordre chronologique avec le format: "Lundi 15 janvier 2024"
  
  Question: "{question}"
  
  Répondez maintenant comme si vous parliez à un étudiant ou visiteur devant vous, de manière naturelle et utile:`;

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
  Si l'utilisateur vous salue ou bien le context contenient un mot (comme "bonjour", "hello", etc.), répondez avec ce message "Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?".
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

  // console.log(rephraseQuestionChain);

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
        configurable: { sessionId: `${message.From}` },
      }
    );

    await sendMessage(message.From, answer);
    const newResposne = await vectorStore.similaritySearch(message.Body)

    console.log(message.Body);
    console.log("************************");
    console.log(newResposne);




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
