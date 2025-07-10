import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { embeddings } from "./openai";

import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";

export const vectorStore = new MemoryVectorStore(embeddings);
// const vectorStore = await PGVectorStore.initialize(embeddings, {});

export const retriever = vectorStore.asRetriever();