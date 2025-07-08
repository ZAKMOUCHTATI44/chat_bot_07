import express from "express";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm } from "./src/openai";
import { retriever, vectorStore } from "./src/vector";
import cors from "cors";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import * as parse from "pdf-parse";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize FAQ data and vector store
let faqGraph: any;

async function initializeFAQSystem() {
  // Load FAQ data from JSON file

  const data = new JSONLoader("./data/data.json");
  const loader = new JSONLoader("./data/data.json");
  const JSONLoaderConcours = new JSONLoader("./data/concours.json");
  const pdf = new PDFLoader("./data/Catalogue.pdf");

  const docs = await loader.load();
  const pdfload = await pdf.load();
  const dataload = await data.load();
  const concoursLoad = await JSONLoaderConcours.load();
  // console.log("Loaded  documents:", pdfload.length);
  // const splitter = new RecursiveCharacterTextSplitter({
  //   chunkSize: 1000,
  //   chunkOverlap: 200,
  // });
  // const allSplits = await splitter.splitDocuments(docs);

  // Index chunks
  await vectorStore.addDocuments(docs);
  await vectorStore.addDocuments(pdfload);
  await vectorStore.addDocuments(concoursLoad);
  await vectorStore.addDocuments(dataload);

  // Define prompt for question-answering
  const template = `Vous êtes l'assistant de l'Université Internationale de Rabat. Répondez poliment et professionnellement. 
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
app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const inputsQA = { question };
    let finalAnswer = "";

    for await (const chunk of await faqGraph.stream(inputsQA, {
      streamMode: "updates",
    })) {
      if (chunk.generate?.answer) {
        finalAnswer = chunk.generate.answer;
      }
    }

    res.json({ question, answer: finalAnswer });
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
