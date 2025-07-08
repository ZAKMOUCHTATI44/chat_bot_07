import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { embeddings } from "./openai";

export const vectorStore = new MemoryVectorStore(embeddings);



export const retriever = vectorStore.asRetriever();



// Initialize the ensemble retriever
// const ensembleRetriever = new EnsembleRetriever({
//     retrievers: [bm25Retriever, vectorStoreRetriever],
//     weights: [0.5, 0.5],
//   });