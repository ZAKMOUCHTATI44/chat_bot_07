import { CustomChatModel } from "./src/CustomChatModel";

const main = async () => {
  const chatModel = new CustomChatModel({ n: 10 });

  const res = await chatModel.invoke([["UIR", "what is the UIR "]]);

  console.log(res);
};

main();
