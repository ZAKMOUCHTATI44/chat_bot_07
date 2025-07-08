// import {
//   START,
//   END,
//   MessagesAnnotation,
//   StateGraph,
//   MemorySaver,
// } from "@langchain/langgraph";
// import llm from "./openai";

// // Define the function that calls the model
// const callModel = async (state: typeof MessagesAnnotation.State) => {
//   const response = await llm.invoke(state.messages);
//   return { messages: response };
// };

// // Define a new graph
// const workflow = new StateGraph(MessagesAnnotation)
//   // Define the node and edge
//   .addNode("model", callModel)
//   .addEdge(START, "model")
//   .addEdge("model", END);

// // Add memory
// const memory = new MemorySaver();
// const app = workflow.compile({ checkpointer: memory });

// console.log(app)

// export default app;
