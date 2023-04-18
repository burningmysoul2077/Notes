> This guide covers the basics of prompts to provide a rough idea of how to use prompts to interact and instruct large language models (LLMs).
> All examples are tested with `text-davinci-003` (using OpenAI's playground) unless otherwise specified. It uses the default configurations, i.e., `temperature=0.7` and `top-p=1`

---

# LLM Settings 大语言模型的设置

>  When working with prompts, you will be interacting with the LLM via an API or directly. You can configure a few parameters to get different results for your prompts.

-  当使用提示(prompts)时，您将通过一个API或是直接与大语言模型进行交互。
-  为了得到提示的不同结果，您可以设置一些参数。

> **Temperature** - In short, the lower the `temperature` the more deterministic the results in the sense that the highest probable next token is always picked. Increasing temperature could lead to more randomness encouraging more diverse or creative outputs. We are essentially increasing the weights of the other possible tokens. In terms of application, we might want to use a lower temperature value for tasks like fact-based QA to encourage more factual and concise responses. For poem generation or other creative tasks, it might be beneficial to increase the temperature value.

-  **Temperature**，简而言之，temperature越低，结果就越确定，某种意义上是说，总是选中下一个可能性最高的字词(token)
-  增加 temperature 会导致随机性更强，后果就是鼓励更宽泛或更具创造力的输出。这么做实际上，我们是在增加其他可能性的字词的权重(weights)
-  在实际程序应用方面，对于基于事实的问答(QA)，我们需要使用较低的 temperature 来鼓励更真实简洁的回应；而对于像是诗歌生成或其他创造性任务，提高 temperature 值也许是更有益的。

>  **Top_p** - Similarly, with `top_p`, a sampling technique with temperature called nucleus sampling, you can control how deterministic the model is at generating a response. If you are looking for exact and factual answers keep this low. If you are looking for more diverse responses, increase to a higher value.

-  **Top_p**，也可以是 **top_p**，一种称为核采样(nucleus sampling)的temperature采样技术，您可以通过它来控制模型生成一个回应的确定性
-  如果您想要答案真实准确，就把这个参数设为比较低的值
-  如果您想要得到更宽泛的回应，就将参数设为比较高的值

>  The general recommendation is to alter one, not both.

-  普遍推荐的做法是这两个参数同时间只改变一个，不要两个都同时改变

>  Before starting with some basic examples, keep in mind that your results may vary depending on the version of LLM you are using.

-  在我们开始一些简单例子之前，请牢记您的结果会因为大语言模型的版本不同而有所差异。

---

# Basic of Prompting 提示基础

## Basic Prompts 基础的提示

>  You can achieve a lot with simple prompts, but the quality of results depends on how much information you provide it and how well-crafted it is. A prompt can contain information like the _instruction_ or _question_ you are passing to the model and including other details such as _context_, _inputs_, or _examples_. You can use these elements to instruct the model better and as a result get better results.

>  Let's get started by going over a basic example of a simple prompt:

-  您可以用简单的提示就完成一些任务，但是得到的结果质量完全取决于您提供给模型的信息量及其精细程度(well-crafted)。提示可以包含信息，就比如您传递给模型的 *指令(instruction)* 或是 *问题(question)*，还可以包括细节，比如 *上下文(context) 输入(inputs)* 或 *示例(examples)*。您完全可以使用这些元素来指导模型，从而获得更好的结果。
-  让我们以一个简单提示的基本例子作为开始

 _Prompt_

```
The sky is
```

_Output:_

```
blueThe sky is blue on a clear day. On a cloudy day, the sky may be gray or white.
```

>  As you can see, the language model outputs a continuation of strings that make sense given the context `"The sky is"`. The output might be unexpected or far from the task we want to accomplish.

>  This basic example also highlights the necessity to provide more context or instructions on what specifically we want to achieve.

-  由上可见，在给定“The sky is”作为前文的情况下，语言模型输出了一系列的还算讲得通的字符串。但输出可能是意想不到的、甚至是与我们想要的风牛马不相及。
-  这个基础示例同样强调了：我们还需要根据具体的实现目标来提供更多的上下文和指令。

>  Let's try to improve it a bit:

_Prompt:_

```
Complete the sentence: The sky is
```

_Output:_

```
so  beautiful today.
```

>  Is that better? Well, we told the model to complete the sentence so the result looks a lot better as it follows exactly what we told it to do ("complete the sentence"). This approach of designing optimal prompts to instruct the model to perform a task is what's referred to as **prompt engineering**.

>  The example above is a basic illustration of what's possible with LLMs today. Today's LLMs are able to perform all kinds of advanced tasks that range from text summarization to mathematical reasoning to code generation.

-  是不是好了一点？好吧，这是因为我们告诉模型要去怎样完成句子，而它也确实按照我们说的做了。这种通过设计最佳提示来指导模型执行某种任务的方法，就被称为**提示工程 (Prompt Engineering)**
-  上面的示例基本说明了当今大语言模型的能力范围。当今的大语言模型，可以说是能够执行各种各样的高级任务，从文本摘要、到数学推理、再到代码生成等。

## Prompt Formatting 格式化提示

>  We have tried a very simple prompt above. A standard prompt has the following format:

-  我们上面尝试了一个非常简单的提示。一个标准的提示有着如下的格式：

```
<Question>?
```

>  or

```
<Instruction>
```

>  This can be formatted into a question answering (QA) format, which is standard in a lot of QA datasets, as follows:

-  这个可以被格式化成问答(QA)形式，这也是问答数据集中的标准格式，如下所示：

```
Q: <Question>?
A: 
```

>  When prompting like the above, it's also referred to as _zero-shot prompting_, i.e., you are directly prompting the model for a response without any examples or demonstrations about the task you want it to achieve. Some large language models do have the ability to perform zero-shot prompting but it depends on the complexity and knowledge of the task at hand.

-  当出现像上面示例这样的提示，它被称为 **零样本提示(zero-shot prompting)**，即，您不需要提供任何关于要实现的任务的示例或是演示，而是直接提示模型得到回应，。一些大型语言模型确实具有执行零样本提示的能力，不过这也取决于模型手头任务的复杂性和知识度。

>  Given the standard format above, one popular and effective technique to prompting is referred to as _few-shot prompting_ where we provide exemplars (i.e., demonstrations). Few-shot prompts can be formatted as follows:

-  根据上面的标准格式，一种流行而又有效的提示技术被叫做 **少样本提示(few-shot prompting)**，这种技术需要我们提供示例(演示)。少样本提示的格式化如下：

```
<Question>?
<Answer>
<Question>?
<Answer>
<Question>?
<Answer>
<Question>?
```

>  The QA format version would look like this:

-  QA版本的格式化如下：

```
Q: <Question>?
A: <Answer>
Q: <Question>?
A: <Answer>
Q: <Question>?
A: <Answer>
Q: <Question>?
A:
```

>  Keep in mind that it's not required to use QA format. The prompt format depends on the task at hand. For instance, you can perform a simple classification task and give exemplars that demonstrate the task as follows:

-  请谨记您不是必须使用QA格式。提示得格式完全取决于手头的任务。
-  例如，您可以执行一个简单的分类任务，并通过如下的示例来演示该如何完成任务:

_Prompt:_

```
This is awesome! // Positive
This is bad! // Negative
Wow that movie was rad! // Positive
What a horrible show! //
```

_Output:_

```
Negative
```

>  Few-shot prompts enable in-context learning which is the ability of language models to learn tasks given a few demonstrations.

-  少样本提示提供了文本内学习，这也是语言模型通过一些演示就能学习任务的能力。

---

# Prompt Elements 提示的元素

## Elements of a Prompt 一个提示所包含的元素

>  As we cover more and more examples and applications that are possible with prompt engineering, you will notice that there are certain elements that make up a prompt.

>  A prompt can contain any of the following components:

>  **Instruction** - a specific task or instruction you want the model to perform

>  **Context** - can involve external information or additional context that can steer the model to better responses

>  **Input Data** - is the input or question that we are interested to find a response for

>  **Output Indicator** - indicates the type or format of the output.

>  Not all the components are required for a prompt and the format depends on the task at hand. We will touch on more concrete examples in upcoming guides.

-  随着我们介绍的提示工程的示例和应用越多，您会注意到组成一个提示有几个固定元素。
-  一个提示可以包含以下任何一部分：
	-  **指令(Instruction)** ： 您想要模型执行的一个特定任务或指令
	-  **上下文(Context)**： 可以包括外部信息或引导模型做出更好回应的附加上下文
	-  **输入数据(Input Data)**：可以是您的输入，也可以是您感兴趣想要得到回应的问题
	-  **输出指示(Output Indicator)**：指示输出的类型和格式
-  一个提示不必包含以上所有部分，而且格式取决于手头的任务。
-  接下来，我们会接触到更多的实例。

---

# General Tips for Designing Prompts 设计提示的技巧

>  Here are some tips to keep in mind while you are designing your prompts:

-  下面是您再设计您的提示也许会用到的一些技巧。

## Start Simple 开始时简单点

>  As you get started with designing prompts, you should keep in mind that it is really an iterative process that requires a lot of experimentation to get optimal results. Using a simple playground like OpenAI or Cohere's is a good starting point.

-  在开始设计提示之前，请牢记，这实际上是一个迭代的过程，需要进行大量大量的实验才有可能能获得最佳结果。像是使用 OpenAI 或 Cohere 就是一个很好的起点。

> You can start with simple prompts and keep adding more elements and context as you aim for better results. Versioning your prompt along the way is vital for this reason. As we read the guide you will see many examples where specificity, simplicity, and conciseness will often give you better results.

-  您可以从一些简单的提示开始，为了得到更好的结果就不断地向其增加更多的元素和上下文。为此，对提示进行版本化就显得相当重要。当我们阅读这份指南时，您会看到许许多多使用具体、朴素和简洁提示却能得到好的结果的例子。

>  When you have a big task that involves many different subtasks, you can try to break down the task into simpler subtasks and keep building up as you get better results. This avoids adding too much complexity to the prompt design process at the beginning.

-  比方说，您有一个大任务，而这个大任务又牵连到不同子任务，您可以尝试把大任务分解成若干个更简单的子任务，然后慢慢累积起来，直到得到想要的结果。这样就避免了在一开始就把太高的复杂度附加在提示设计过程上。

## The Instruction 指令

>  You can design effective prompts for various simple tasks by using commands to instruct the model what you want to achieve such as "Write", "Classify", "Summarize", "Translate", "Order", etc.

-  您可以使用命令(commands)为各种的简单任务设计有效的提示，以便指导模型实现您想要的任务，命令可以是
	-  “编写 Write”
	-  “分类 Classify”
	-  “总结 Summarize”
	-  “翻译 Translate”
	-  “下令 Order” 等。

>  Keep in mind that you also need to experiment a lot to see what works best. Try different instructions with different keywords, contexts, and data and see what works best for your particular use case and task. Usually, the more specific and relevant the context is to the task you are trying to perform, the better. We will touch on the importance of sampling and adding more context in the upcoming guides.

>  Others recommend that instructions are placed at the beginning of the prompt. It's also recommended that some clear separator like "###" is used to separate the instruction and context.

-  请记住，您需要进行大量的试验才能发现哪种指令最有效。要尝试使用不同的指令混合不同的关键字、上下文和数据，看看哪种方法最适合您的特定用例和任务。通常情况下，想要效果越好，上下文就需要与试图执行的任务越具体、越相关。在即将发布的指南中，我们将提及采样和添加更多上下文的必要性。
-  还有，建议将指令放在提示的最开头。此外，还建议使用清晰的分隔符，如“###”，来分隔指令和上下文。

For instance:

_Prompt:_

```
### Instruction ###
Translate the text below to Spanish:
Text: "hello!"
```

_Output:_

```
¡Hola!
```

## Specificity 专一性

>  Be very specific about the instruction and task you want the model to perform. The more descriptive and detailed the prompt is, the better the results. This is particularly important when you have a desired outcome or style of generation you are seeking. There aren't specific tokens or keywords that lead to better results. It's more important to have a good format and descriptive prompt. In fact, providing examples in the prompt is very effective to get desired output in specific formats.

-  要非常具体地说明您所希望模型执行的指令和任务。提示的描述性和细节性越详细，结果肯定就越如意。当您希望有一个理想的结果或是有一种一直在寻找的生成体裁时，这一点可谓是极其重要。死心吧，是不会有特定的字词或关键字就能带来好的结果的。相比之下，更重要的是，要有一个好的格式和描述性的提示。事实上，在提示中提供示例对于以特定格式获得理想的输出这类任务是非常有效的。

>  When designing prompts you should also keep in mind the length of the prompt as there are limitations regarding how long this can be. Thinking about how specific and detailed you should be is something to consider. Including too many unnecessary details is not necessarily a good approach. The details should be relevant and contribute to the task at hand. This is something you will need to experiment with a lot. We encourage a lot of experimentation and iteration to optimize prompts for your applications.

-  请牢记，在设计提示时，一定要非常注意提示的长度限制。将提示考虑的具体周祥是应该的，但包含太多不必要的细节并不是一个好方法，细节应该是和手头的任务相关的、有帮助的，在这点上，您需要大量的试验。我们也鼓励通过大量的试验和迭代来优化应用程序的提示。

>  As an example, let's try a simple prompt to extract specific information from a piece of text.

-  让我们尝试从一个简单的提示示例开始：我们想要从一段文本中提取特定信息。

_Prompt:_

```
Extract the name of places in the following text. 
Desired format:
Place: <comma_separated_list_of_company_names>
Input: "Although these developments are encouraging to researchers, much is still a mystery. “We often have a black box between the brain and the effect we see in the periphery,” says Henrique Veiga-Fernandes, a neuroimmunologist at the Champalimaud Centre for the Unknown in Lisbon. “If we want to use it in the therapeutic context, we actually need to understand the mechanism.""
```

_Output:_

```
Place: Champalimaud Centre for the Unknown, Lisbon
```

>  Input text is obtained from [this Nature article(opens in a new tab)](https://www.nature.com/articles/d41586-023-00509-z).

## Avoid Impreciseness 避免含义模糊

>  Given the tips above about being detailed and improving format, it's easy to fall into the trap of wanting to be too clever about prompts and potentially creating imprecise descriptions. It's often better to be specific and direct. The analogy here is very similar to effective communication -- the more direct, the more effective the message gets across.

-  考虑到上个小节提出的关于细节和改进格式的建议，这很容易陷入另一个陷阱。这是因为如果在提示上花费太多想法，那样创造出来的描述反而不精确，所以别想太多，既具体又直接通常会表现的更好。这与有效沟通相当类似——与其弯弯绕绕不如单刀直入。

>  For example, you might be interested in learning the concept of prompt engineering. You might try something like:

-  比如说，您对学习提示工程的概念感兴趣。您进行如下尝试：

```
Explain the concept prompt engineering. Keep the explanation short, only a few sentences, and don't be too descriptive.
```

>  It's not clear from the prompt above how many sentences to use and what style. You might still somewhat get good responses with the above prompts but the better prompt would be one that is very specific, concise, and to the point. Something like:

-  上面的提示没有明确指出该用多少句话、用什么体裁。虽然使用上述提示，可能得到的回复还不错，但更明确的提示应该是非常具体、简洁、开门见山的。就像下面这样：

```
Use 2-3 sentences to explain the concept of prompt engineering to a high school student.
```

## To do or not to do? 该做什么或不该做什么

>  Another common tip when designing prompts is to avoid saying what not to do but say what to do instead. This encourages more specificity and focuses on the details that lead to good responses from the model.

-  设计提示还有一个常见的技巧：避免说“不该做什么 (what not to do)”，改为说 “该做什么(what to do)”。这样设计会鼓励更专一、更关注那些帮助模型做出更好回应的细节。

>  Here is an example of a movie recommendation chatbot failing at exactly what I don't want it to do because of how I wrote the instruction -- focusing on what not to do.

-  下面有个聊天机器人推荐电影的例子，它不幸的踩雷了，正是由于我写的指令——专注于不该做什么

_Prompt:_

```
The following is an agent that recommends movies to a customer. DO NOT ASK FOR INTERESTS. DO NOT ASK FOR PERSONAL INFORMATION.
Customer: Please recommend a movie based on my interests.
Agent: 
```

_Output:_

```
Sure, I can recommend a movie based on your interests. What kind of movie would you like to watch? Do you prefer action, comedy, romance, or something else?
```

>  Here is a better prompt:

-  下面这个提示就表现得更好：

_Prompt:_

```
The following is an agent that recommends movies to a customer. The agent is responsible to recommend a movie from the top global trending movies. It should refrain from asking users for their preferences and avoid asking for personal information. If the agent doesn't have a movie to recommend, it should respond "Sorry, couldn't find a movie to recommend today.".
Customer: Please recommend a movie based on my interests.
Agent:
```

_Output:_

```
Sorry, I don't have any information about your interests. However, here's a list of the top global trending movies right now: [list of movies]. I hope you find something you like!
```

>  Some of the examples above were adopted from the ["Best practices for prompt engineering with OpenAI API" article.](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

---

# Examples of Prompts 提示示例

>  In the previous section, we introduced and gave a basic examples of how to prompt LLMs.

>  In this section, we will provide more examples of how prompts are used to achieve different tasks and introduce key concepts along the way. Often, the best way to learn concepts is by going through examples. Below we cover a few examples of how well-crafted prompts can be used to perform different types of tasks.

-  在前一节中，我们介绍并给出了如何提示大语言模型的基础示例。
-  在本节中，我们将提供更多关于如何使用提示来完成不同任务的示例，并在此过程中介绍关键的概念。学习概念通常的最好方法是通过示例学习。下面我们将介绍几个示例，说明如何使用精心设计的提示来执行不同类型的任务。

Topics:

-   [Text Summarization](#text-summarization-文本总结)
-   [Information Extraction]
-   [Question Answering]
-   [Text Classification]
-   [Conversation]
-   [Code Generation]
-   [Reasoning]

---

## Text Summarization 文本摘要

>  One of the standard tasks in natural language generation is text summarization. Text summarization can include many different flavors and domains. In fact, one of the most promising applications of language models is the ability to summarize articles and concepts into quick and easy-to-read summaries. Let's try a basic summarization task using prompts.

-  自然语言生成的标准任务之一是文本摘要，它可以包括许多不同的风格和领域。实际上，语言模型最有前途的应用之一就是：能够将文章和思想总结成易于速读的摘要。让我们用提示来做一个摘要任务。

>  Let's say I am interested to learn about antibiotics, I could try a prompt like this:

-  比方说我对抗生素感兴趣，我可以试着使用下面这样的提示：

_Prompt:_

```
Explain antibiotics
A:
```

_Output:_

```
Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.
```

>  The "A:" is an explicit prompt format that's used in question answering. I used it here to tell the model that there is an expected further. In this example, it's not clear how this is useful vs not using it but we will leave it that for later examples. Let's just assume that this is too much information and want to summarize it further. In fact, we can instruct the model to summarize into one sentence like so:

-  "A:"， 这个是用在问答中显式的提示格式。在这里，我用来告诉模型需要进一步的解释。在这个示例中，尚不清楚使用它比不使用它强在哪，我们把这个问题留到以后的示例中。让我们假设这里面有太多的信息，需要进一步来总结。实际上，我们可以指导模型去像下面这样总结成一句话：

_Prompt:_

```
Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.Explain the above in one sentence:
```

_Output:_

```
Antibiotics are medications used to treat bacterial infections by either killing the bacteria or stopping them from reproducing, but they are not effective against viruses and overuse can lead to antibiotic resistance.
```

>  Without paying too much attention to the accuracy of the output above, which is something we will touch on in a later guide, the model tried to summarize the paragraph in one sentence. You can get clever with the instructions but we will leave that for a later chapter. Feel free to pause here and experiment to see if you get better results.

-  我们没有过多关心上面输出的准确性，这个话题将在后面的指南中讨论。模型试图仅用一句来概括这整个一段话。您可以使用这些指令，但到后面的章节我们还会讨论。请在此暂停一会，自己试验一下，看看您能否得到更好的结果。

## Information Extraction 信息抽取

>  While language models are trained to perform natural language generation and related tasks, it's also very capable of performing classification and a range of other natural language processing (NLP) tasks.

-  虽然语言模型被训练来执行自然语言生成及其相关任务，但它也能够执行分类和一系列其他的自然语言处理任务。

>  Here is an example of a prompt that extracts information from a given paragraph.

-  下面示例是：使用提示来从给定文本段落中做信息抽取。

_Prompt:_

```
Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.Mention the large language model based product mentioned in the paragraph above:
```

_Output:_

```
The large language model based product mentioned in the paragraph above is ChatGPT.
```

>  There are many ways we can improve the results above, but this is already very useful.

-  我们当然还有很多方法来改善上面的结果，但实话说这结果已经很好了。

>  By now it should be obvious that you can ask the model to perform different tasks by simply instructing it what to do. That's a powerful capability that AI product developers are already using to build powerful products and experiences.

-  目前为止的示例表现的很明显，可以通过简单地指示模型去做什么来要求它执行不同的任务。AI产品开发人员已经在运用这种能力来帮助构建强大的产品和用户体验。

Paragraph source: [ChatGPT: five priorities for research(opens in a new tab)](https://www.nature.com/articles/d41586-023-00288-7)

## Question Answering 问答系统

>  One of the best ways to get the model to respond to specific answers is to improve the format of the prompt. As covered before, a prompt could combine instructions, context, input, and output indicators to get improved results. While these components are not required, it becomes a good practice as the more specific you are with instruction, the better results you will get. Below is an example of how this would look following a more structured prompt.

-  能够让模型对特定答案做出回应的最好方法之一就是：改进提示的格式。如前文所述，一个提示可以包含指令、上下文、输入和输出指示。虽然这些组件不是必需的，但由于您指令给的越具体结果就越好而变成一种好的实践。下面用一个示例来说明一个更有条理的提示会产生什么结果。

_Prompt:_

```
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.
Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
Question: What was OKT3 originally sourced from?
Answer:
```

_Output:_

```
Mice.
```

Context obtained from [Nature(opens in a new tab)](https://www.nature.com/articles/d41586-023-00400-x).

## Text Classification 文本分类

>  So far, we have used simple instructions to perform a task. As a prompt engineer, you will need to get better at providing better instructions. But that's not all! You will also find that for harder use cases, just providing instructions won't be enough. This is where you need to think more about the context and the different elements you can use in a prompt. Other elements you can provide are `input data` or `examples`.

-  到目前为止，我们都是在用简单的指令来执行任务。作为一名提示工程师，您需要想出更好的指令，但这远远不是全部！您还会发现对于更难的用例，仅提供说明是远远不够的。这就是说，您需要更多地考虑上下文或是其他可以用在提示中的不同元素，可以是“输入数据(input data)”或“示例(examples)”。

>  Let's try to demonstrate this by providing an example of text classification.

-  让我们用一个文本分类的示例来做说明吧

_Prompt:_

```
Classify the text into neutral, negative or positive. 
Text: I think the food was okay. 
Sentiment:
```

_Output:_

```
Neutral
```

>  We gave the instruction to classify the text and the model responded with `'Neutral'` which is correct. Nothing is wrong with this but let's say that what we really need is for the model to give the label in the exact format we want. So instead of `Neutral` we want it to return `neutral`. How do we achieve this? There are different ways to do this. We care about specificity here, so the more information we can provide the prompt the better results. We can try providing examples to specify the correct behavior. Let's try again:

-  我们下达了对文本进行分类的指令，模型给出回答是 'Neutral'，完全正确。表面上并没有问题，但是我们想让模型用确切的格式回答出标签。所以，我们希望它返回的是 'neutral' 而不是 'Neutral'。我们是如何做到这一点的？是有很多不同方法的。在这里我们关心的是专一性，所以能提供给提示的信息越多，结果就越好。我们可以试着提供示例来特别指明正确的行为。那我们再试试:

_Prompt:_

```
Classify the text into neutral, negative or positive. 
Text: I think the vacation is okay.
Sentiment: neutral 
Text: I think the food was okay. 
Sentiment:
```

_Output:_

```
neutral
```

>  Perfect! This time the model returned `neutral` which is the specific label I was looking for. It seems that the example provided in the prompt helped the model to be specific in its output.

-  完美！这次模型返回了 'neutral'，这正是我在寻找的特定标签。提供在提示中的示例似乎对于模型的输出变得更加具体起到了很大帮助。

>  To highlight why sometimes being specific is important, check out the example below and spot the problem:

-  为了强调为什么有时候具体是重要的。来看看下面的示例，试着找出不同:

_Prompt:_

```
Classify the text into nutral, negative or positive. 
Text: I think the vacation is okay.
Sentiment:
```

_Output:_

```
Neutral
```

>  What is the problem here? As a hint, the made up `nutral` label is completely ignored by the model. Instead, the model outputs `Neutral` as it has some bias towards that label. But let's assume that what we really want is `nutral`. How would you fix this? Maybe you can try adding descriptions to the labels or add more examples to the prompt? If you are not sure, we will discuss a few ideas in the upcoming sections.

-  发现问题在哪了嘛？我们编造了 'nutral' 标签作为暗示，但完全被模型忽略了。相反，模型输出了带有偏向的 'Neutral'。但是，如果假设我们真正想要的是 'nutral'，该如何解决这个问题呢？也许您可以尝试在标签中添加更多的描述？或是在提示中添加更多的示例？如果您还不确定，在接下来的部分中我们会进行讨论。

## Conversation 对话系统

>  Perhaps one of the more interesting things you can achieve with prompt engineering is instructing the LLM system on how to behave, its intent, and its identity. This is particularly useful when you are building conversational systems like customer service chatbots.

-  也许通过提示工程可以实现的比较有趣的事情之一是指导大语言模型如何表现它的意图和身份。如果您想要构建对话系统(如客服聊天机器人)时特别有用。

>  For instance, let's create a conversational system that's able to generate more technical and scientific responses to questions. Note how we are explicitly telling it how to behave through the instruction. This is sometimes referred to as _role prompting_.

-  举例来说，我们创造一个对话系统，它能够生成兼具技术与科学的回答。请注意我们是如何通过指令显式地告诉它如何去做的的，这被称为 __角色提示(role prompting)__。

_Prompt:_

```
The following is a conversation with an AI research assistant. The assistant tone is technical and scientific.
Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of blackholes?
AI:
```

_Output:_

```
Sure! Black holes are regions of spacetime where the gravitational force is so strong that nothing, not even light, can escape from it. They are created when a very massive star dies and its core collapses in on itself, forming a singularity of infinite density. The intense gravity of the singularity pulls in all the matter and radiation around it, creating the black hole.
```

>  Our AI research assistant sounds a bit too technical, right? Okay, let's change this behavior and instruct the system to give more accessible answers.

-  听起来我们的AI研究助理有点太过于专业了吧，不是吗？。好吧，让我们小小改变一下它的行为，指导系统给出更容易理解的答案。

_Prompt:_

```
The following is a conversation with an AI research assistant. The assistant answers should be easy to understand even by primary school students.
Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of black holes?
AI: 
```

_Output:_

```
Sure! Black holes are created when a very massive star runs out of fuel and collapses in on itself. The gravity of the star is so strong that nothing, not even light, can escape its pull. This creates a point in space-time with an intense gravitational force called a black hole.
```

>  I think we made some progress. You can continue improving it. I am sure if you add more examples you might get even better results.

-  我想我们已经有所进步。您可以继续改善它。我十分确信如果您继续增加更多的示例，一定还会有所进步。

## Code Generation 代码生成

>  One application where LLMs are quite effective is code generation. Copilot is a great example of this. There are a vast number of code-generation tasks you can perform with clever prompts. Let's look at a few examples below.

-  代码生成就是大语言模型能产生非常有效作用的一种应用。Copilot就是一个很好的例子。您可以使用巧妙的提示来执行大量的代码生成任务。让我们看看下面的几个示例。

>  First, let's try a simple program that greets the user.

-  首先，我们来试一个欢迎用户的简单程序

_Prompt:_

```
/*
Ask the user for their name and say "Hello"
*/
```

_Output:_

```
let name = prompt("What is your name?");
console.log(`Hello, ${name}!`);
```

>  You can see that we didn't even need to specify the language to use.

>  Let's switch levels a bit. I want to show you how powerful LLMs can be with a little more effort in designing the prompts.

-  您可以看到我们甚至不需要指定所要使用的语言。
-  让我们提升一下难度。我想向您展示大语言模型在设计提示时所表现出的功能之强大。

_Prompt:_

```
"""
Table departments, columns = [DepartmentId, DepartmentName]
Table students, columns = [DepartmentId, StudentId, StudentName]
Create a MySQL query for all students in the Computer Science Department
"""
```

_Output:_

```
SELECT StudentId, StudentName 
FROM students 
WHERE DepartmentId IN (SELECT DepartmentId FROM departments WHERE DepartmentName = 'Computer Science');
```

>  This is very impressive. In this case, we provided data about the database schema and asked it to generate a valid MySQL query.

-  这让人印象深刻。在这个用例中，我们提供了关于数据库模式的数据，并要求他生成一条可执行的MySQL查询语句。

## Reasoning 推理

>  Perhaps one of the most difficult tasks for an LLM today is one that requires some form of reasoning. Reasoning is one of the areas that I am most excited about due to the types of complex applications that can emerge from LLMs.

>  There have been some improvements in tasks involving mathematical capabilities. That said, it's important to note that current LLMs struggle to perform reasoning tasks so this requires even more advanced prompt engineering techniques. We will cover these advanced techniques in the next guide. For now, we will cover a few basic examples to show arithmetic capabilities.

-  某种意义上，当今大语言模型遇到的最困难的任务之一，就是那种需要某种形式的推理的任务。推理也是我最感兴趣的领域之一，因为大语言模型可以涌现出复杂的应用类型。

-  对于涉及到数学能力的任务已经有了一些改进。也就是需要注意的是，当前的大语言模型难以执行推理任务，因此它需要更先进的提示工程技术。下一篇指南中，我们将会介绍这些进阶技术。目前，我们将介绍几个基本示例来展示一下算术能力。

_Prompt:_

```
What is 9,000 * 9,000?
```

_Output:_

```
81,000,000
```

>  Let's try something more difficult.

-  试个更难的。

_Prompt:_

```
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A: 
```

_Output_

```
No, the odd numbers in this group add up to an odd number: 119.
```

>  That's incorrect! Let's try to improve this by improving the prompt.

-  可惜不对呀！试试改进一下提示。

_Prompt:_

```
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
Solve by breaking the problem into steps. First, identify the odd numbers, add them, and indicate whether the result is odd or even. 
```

_Output:_

```
Odd numbers: 15, 5, 13, 7, 1
Sum: 41 
41 is an odd number.
```

>  Much better, right? By the way, I tried this a couple of times and the system sometimes fails. If you provide better instructions combined with examples, it might help get more accurate results.

-  怎么样？好多了吧。顺便说一句，我试了很多遍，当中甚至有几次弄到系统崩溃。如果您能够结合示例提供更好的指令，那将会对得到更准确的结果大有裨益。

>  We will continue to include more examples of common applications in this section of the guide.

>  In the upcoming section, we will cover even more advanced prompt engineering concepts and techniques for improving performance on all these and more difficult tasks.

-  在本部分中，我们将继续包括更多常见应用程序的示例。
-  在下一节中，我们将介绍更高级的提示工程概念和技术，以提高性能来对付所有这些、甚至更困难的任务。
