import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";

require("dotenv").config();

export const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  // model: "gpt-4o-mini",
  modelName: "gpt-3.5-turbo-1106",
  // temperature: 0,
});

export const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});
