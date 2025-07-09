import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { embeddings } from "./openai";

import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";

export const vectorStore = new MemoryVectorStore(embeddings);
// const vectorStore = await PGVectorStore.initialize(embeddings, {});

export const retriever = vectorStore.asRetriever();

// Initialize the ensemble retriever
// const ensembleRetriever = new EnsembleRetriever({
//     retrievers: [bm25Retriever, vectorStoreRetriever],
//     weights: [0.5, 0.5],
//   });
