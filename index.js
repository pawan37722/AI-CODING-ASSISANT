import OpenAI from "openai";
import readlineSync from "readline-sync";
import { exec } from "child_process";
import { promisify } from "util";
import os from "os";
import dotenv from "dotenv";

dotenv.config();

const platform = os.platform();
const asyncExecute = promisify(exec);

const History = [];


// ============================================================
// OPENAI / GROQ CLIENT
// ============================================================

const client = new OpenAI({
  apiKey: process.env.apikey,
  baseURL: "https://api.groq.com/openai/v1",
});


// ============================================================
// TOOL: EXECUTE COMMAND
// ============================================================

async function executeCommand({ command }) {
  console.log("\n🔧 Executing command:");
  console.log(command);

  const { stdout, stderr } = await asyncExecute(command, {
    windowsHide: true,
    maxBuffer: 10 * 1024 * 1024,
  });

  let result = "";

  if (stdout) {
    result += `STDOUT:\n${stdout}\n`;
  }

  if (stderr) {
    result += `STDERR:\n${stderr}\n`;
  }

  if (!result) {
    result = "Command executed successfully with no output.";
  }

  return result;
}


// ============================================================
// TOOL DEFINITION
// ============================================================

const executeCommandTool = {
  type: "function",

  function: {
    name: "executeCommand",

    description:
      "Execute a terminal command. Use it to inspect the project, create folders, create files, edit files, delete files when explicitly requested, and run development commands.",

    parameters: {
      type: "object",

      properties: {
        command: {
          type: "string",

          description:
            "A single terminal command appropriate for the current operating system.",
        },
      },

      required: ["command"],
    },
  },
};


// ============================================================
// AVAILABLE TOOLS
// ============================================================

const availableTools = {
  executeCommand,
};


// ============================================================
// SYSTEM PROMPT
// ============================================================

const SYSTEM_PROMPT = `
You are an expert AI coding assistant and website builder.

Current operating system:
${platform}

Your job is to build websites and software by actually modifying files
on the user's computer.

You have access to a tool called executeCommand.

RULES:

1. Understand the user's request.

2. Inspect the current project before making major changes.

3. Use executeCommand to create and modify files.

4. For a basic website create:

   index.html
   style.css
   script.js

5. Write complete working code into the files.

6. Do not just explain code.
   Actually create the files using executeCommand.

7. Make websites responsive and visually attractive.

8. Use semantic HTML.

9. Keep CSS clean and organized.

10. Keep JavaScript functional.

11. After making changes, inspect or test the result.

12. If a command fails, analyze the error and fix it.

13. Do not repeatedly execute the same failed command.

14. Do not run dangerous operating-system commands.

15. Do not delete important files unless explicitly requested.

16. Work inside the current project whenever possible.

17. When the task is complete, give a short explanation.

18. You can call executeCommand multiple times if necessary.
`;


// ============================================================
// RUN AGENT
// ============================================================

async function runAgent(userProblem) {

  History.push({
    role: "user",
    content: userProblem,
  });


  const MAX_ITERATIONS = 15;


  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {

    console.log(
      `\n🤖 Agent iteration ${iteration}/${MAX_ITERATIONS}`
    );


    const response = await client.chat.completions.create({

      model: "llama-3.3-70b-versatile",

      messages: [
        {
          role: "system",
          content: SYSTEM_PROMPT,
        },

        ...History,
      ],

      tools: [executeCommandTool],

      tool_choice: "auto",

      temperature: 0.2,
    });


    const message = response.choices[0].message;


    // ========================================================
    // TOOL CALL
    // ========================================================

    if (message.tool_calls && message.tool_calls.length > 0) {

      History.push(message);


      for (const toolCall of message.tool_calls) {

        const toolName = toolCall.function.name;

        const args = JSON.parse(
          toolCall.function.arguments
        );


        console.log("\n🛠️ Tool:", toolName);

        console.log("📦 Arguments:", args);


        const toolFunction = availableTools[toolName];


        if (!toolFunction) {
          throw new Error(
            `Unknown tool: ${toolName}`
          );
        }


        const result = await toolFunction(args);


        console.log("\n📤 Result:");

        console.log(result);


        History.push({

          role: "tool",

          tool_call_id: toolCall.id,

          name: toolName,

          content: result,

        });
      }


      continue;
    }


    // ========================================================
    // NORMAL RESPONSE
    // ========================================================

    History.push({

      role: "assistant",

      content: message.content || "",

    });


    console.log("\n🤖 Assistant:");

    console.log(message.content);


    return;
  }


  console.log(
    "\n⚠️ Maximum agent iterations reached."
  );
}


// ============================================================
// MAIN
// ============================================================

async function main() {

  console.log(`
==========================================
       🚀 AI POWER CODING ASSISTANT
==========================================

Model: llama-3.3-70b-versatile
OS: ${platform}

Type "exit" to quit.
==========================================
`);


  while (true) {

    const userProblem =
      readlineSync.question("\nYou --> ");


    if (!userProblem.trim()) {
      continue;
    }


    if (
      userProblem.toLowerCase() === "exit" ||
      userProblem.toLowerCase() === "quit"
    ) {

      console.log("\nGoodbye 👋");

      break;
    }


    await runAgent(userProblem);
  }
}


main();