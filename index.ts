import express from "express";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm } from "./src/openai";
import { initVectorStore } from "./src/vector";
import cors from "cors";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import * as parse from "pdf-parse";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { sendMessage } from "./src/message";
import bodyParser from "body-parser";

const app = express();
const port = process.env.PORT || 7001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Initialize FAQ data and vector store
let faqGraph: any;
let vectorStore: any;

async function initializeFAQSystem() {
  // Load FAQ data from JSON file

  vectorStore = await initVectorStore();

  const data = new JSONLoader("./data/faq.json", ["/question", "/answer"]);
  // const loader = new JSONLoader("./data/data.json");
  const JSONLoaderConcours = new JSONLoader("./data/concours.json");
  const pdf = new PDFLoader("./data/Catalogue.pdf");

  // const docs = await loader.load();
  const pdfload = await pdf.load();
  const dataload = await data.load();
  const concoursLoad = await JSONLoaderConcours.load();
  // console.log("Loaded  documents:", pdfload.length);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(dataload);

  // Index chunks
  await vectorStore.addDocuments(dataload);
  // await vectorStore.addDocuments(pdfload);
  // await vectorStore.addDocuments(concoursLoad);
  // await vectorStore.addDocuments(dataload);

  // Define prompt for question-answering
  const template = `Vous êtes l'assistant de l'Université Internationale de Rabat. Répondez poliment et professionnellement. 
    Si l'utilisateur vous salue ou bien le context contenient un mot (comme "bonjour", "hello", etc.), répondez avec ce message "Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?".
  Si vous ne trouvez pas la réponse dans le contexte, dites 'Je ne sais pas'."


{context}

Question: {question}

Helpful Answer:`;

  const promptTemplate = ChatPromptTemplate.fromMessages([["user", template]]);
  // Define state for application
  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
  });

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  });

  // Define application steps
  const retrieve = async (state: typeof InputStateAnnotation.State) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question);
    return { context: retrievedDocs };
  };

  const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
    });
    const response = await llm.invoke(messages);
    return { answer: response.content };
  };
  // Compile application
  return new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();
}

// API Endpoint to answer questions
app.post("/uir-chat-bot", async (req, res) => {
  try {
    const message = req.body;

    if (!message.Body) {
      res
        .status(400)
        .json({ error: "Question is required in the request body" });
    }
    const inputsQA = { question: message.Body };
    let finalAnswer = "";

    for await (const chunk of await faqGraph.stream(inputsQA, {
      streamMode: "updates",
    })) {
      if (chunk.generate?.answer) {
        finalAnswer = chunk.generate.answer;
      }
    }


    await sendMessage(message.From, finalAnswer);

    res.json({ question : message.Body, answer: finalAnswer });
  } catch (error) {
    console.error("Error processing question:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({ status: "healthy" });
});

// Initialize and start server
initializeFAQSystem()
  .then((graph) => {
    faqGraph = graph;
    app.listen(port, () => {
      console.log(`FAQ API server running on port ${port}`);
    });
  })
  .catch((error) => {
    console.error("Failed to initialize FAQ system:", error);
    process.exit(1);
  });
