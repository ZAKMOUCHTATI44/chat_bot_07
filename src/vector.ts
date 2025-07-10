import {
  PGVectorStore,
  DistanceStrategy,
} from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PoolConfig } from "pg";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const config = {
  postgresConnectionOptions: {
    type: "postgres",
    host: "127.0.0.1",
    port: 5432,
    user: "uir_chat_bot",
    password: "Dv5F0NSl7L1oDRKW3x3N",
    database: "db_uir",
  } as PoolConfig,
  tableName: "testlangchainjs",
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
  distanceStrategy: "cosine" as DistanceStrategy,
};

let vectorStore: PGVectorStore;

export const initVectorStore = async () => {
  vectorStore = await PGVectorStore.initialize(embeddings, config);
  console.log("Inilaizee")
  return vectorStore; // This is what yo
};

// initVectorStore();

export { vectorStore };
