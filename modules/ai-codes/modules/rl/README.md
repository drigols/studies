# Reinforcement Learning

## Contents

 - **Fundaments of Reinforcement Learning:**
   - [Intro to Reinforcement Learning](#intro-to-rl)
   - [The RL components & Process](#rl-components-process)
     - [States](#states)
     - [Actions](#actions)
     - [Rewards](#rewards)
   - [Observations/States Space](#observations-states-space)
   - [Action Space (Discrete vs. Continuous)](#action-space)
   - [Type of tasks (Episodic Task vs. Continuing Tasks)](#types-of-tasks)
   - [Exploitation/Exploration trade-off](#exploitation-exploration-trade-off)
 - [**Settings**](#settings)
 - [**References**](#ref)




































































































<!--- ( Fundaments of Reinforcement Learning ) --->

---

<div id="intro-to-rl"></div>

## Intro to Reinforcement Learning

Well, to start our studies on *Reinforcement Learning*, the first thing you have to keep in mind is that *Reinforcement Learning* is a technique of **"Unsupervised Learning"**.

> In other words, there is no reference or guideline saying that this is the best *action* to take.

Unlike other types of learning, here the only thing the model will have in terms of *supervision (or feedback)* will be a signal that will be the **reward** it obtained in the system and the goal will be to *maximize the reward* as much as possible.

> In other words, I want to gain the maximum possible rewards.

**NOTE:**  
The idea behind (por trás) *Reinforcement Learning* is that an `agent (an AI)` will learn from the *environment* by **interacting with it** (through trial and error) and **receiving rewards** (negative or positive) as feedback to perform actions.

For example, imagine you place your younger brother in front of a video game he's never played before, hand him a controller, and let him play:

![img](images/intro-to-rl-01.png)  

Your brother will interact with the environment (the video game) by pressing the *right button* **(action)**. He has a coin, which is a *reward* of **"+1"**. It's positive, he just understood that in this game **he should collect the coins**...

![img](images/intro-to-rl-02.png)  

So, **he presses to the right again** and touches an enemy, he just died **"-1" reward**...

![img](images/intro-to-rl-03.png)  

 - By interacting with its *environment* through `trial` and `error`, your little brother has just figured out that in this environment **he needs to collect coins**, **but avoid enemies**.
 - Without any supervision, the child will get better at the game over time. That's how humans and animals learn, *through interactions*.

> *Reinforcement Learning* is just a computational approach to learning from *Actions*.










---

<div id="rl-components-process"></div>

## The RL components & Process

The crucial components of **Reinforcement Learning** are:

 - **Environment.**
 - **Agent.**

For example, see the image below:

![img](images/rl-components-process-01.png)  

See that:

 - An **agent** sends *"Actions"* to the **Environment**.
 - And the **environment** sends *"States"* and *Rewards* to the **Agent**.

The **Reinforcement Learning** process is:

![img](images/rl-components-process-02.png)  

See that:

 - First, the **Agent** receives the initial State **"S<sub>t</sub> (or S<sub>0</sub>)"** and Reward **"R<sub>t</sub> (or R<sub>0</sub>)"**.
 - Next, the **Agent** sends the **Action "A<sub>t</sub> (or A<sub>0</sub>)"** to the **Environment**.
 - Finally, the **Environment** sends the State **"S<sub>t+1</sub>"** and Reward **"R<sub>t+1</sub>"** to the **Agent**.

For example, let’s imagine an agent learning to play a platform game:

![img](images/rl-components-process-03.png)

 - Our **Agent** receives **State S<sub>0</sub>** from the **Environment** — We receive the first frame of our game (Environment).
 - Based on that **State S<sub>0</sub>**, the **Agent** takes **Action A<sub>0</sub>** — our Agent will move to the right.
 - The **Environment** goes to a new **State S<sub>1</sub>** — new frame.
 - The **Environment** gives some **Reward R<sub>1</sub>** to the **Agent** — We’re not dead (Positive Reward +1).

**This RL loop outputs a sequence of "State", "Action", "Reward" and next "State":**

![img](images/rl-components-process-04.png)

> The agent’s goal is to `maximize its cumulative reward`, called the expected return.

---

<div id="states"></div>

## States

Well, to begin understanding what **Actions**, **States**, and **Rewards** are, let's think about a *chess game*:

![img](images/states-01.png)

> Let's imagine that the **states** are the positions in which the pieces are located **at the moment**.

If two people (or even one playing against themselves) are playing at a certain moment of the game, the pieces will be positioned in very specific regions of the two opponents, and this is what we can call or define as a **state**.


> **NOTE:**  
> That is, the position in which the pieces are in this specific scenario.

![img](images/states-02.jpg)

> **NOTE:**  
> Well, now you'll agree with me that if I move even a single piece, we'll have another state. That is, another specific scenario.

---

<div id="actions"></div>

## Actions

Okay, but what is an **Action** then? In a chess game, an action can be an agent making a move (moving a piece):

![img](images/Actions-01.gif)

Well, now think with me... What happens with each of my actions in a chess game?

> **A new state is created.**

That's easy to understand because with each **action**, we will have a new *specific scenario*. That is, a new state.

---

<div id="rewards"></div>

## Rewards

> So, following our line of thought in a chess game, the **rewards** will be the **feedback** we will have **after each action**.

Look at the scenario below:

![img](images/rewards-01.gif)

**In this above scenario, was the *reward* *positive* or *negative*?**
Positive (+1) because we moving the red piece (Action) to capture the white piece (the opponent).

> **NOTE:**  
> Another observation here is that an agent can take an **action** that returns *zero*. That is, no pieces were lost on either side.

Now think as if you were a **Data Scientist**, how could you *accumulate* these **rewards** mathematically?

> **With a *Cost Function*.**










---

<div id="observations-states-space"></div>

## Observations/States Space

> *"Observations/States"* are the **information our agent gets from the environment**.

 - In the case of a video game, it can be a frame (a screenshot).
 - In the case of the trading agent, it can be the value of a certain stock, etc.

Briefly, we can think of as:

![img](images/observations-states-space-01.png)










---

<div id="action-space"></div>

## Action Space (Discrete vs. Continuous)

> The **Action Space** is the *set* of **all possible *actions* in an environment (You can think as "Universal Set (U)" or Probability "Sample Space (Ω)")**.

> **NOTE:**  
> The **Actions** can come from a *"discrete"* or *"continuous space"*:

### Discrete Space

> In the case of a **Discrete Space**, the number of possible actions is **finito**.

For example, in *Super Mario Bros*, we have only 4 possible actions:

 - Left.
 - Right.
 - Up (jumping).
 - Down (crouching).

![img](images/action-space-01.png)

**NOTE:**  
Again, in Super Mario Bros, we have a finite set of actions since we have only 4 directions.

### Continuous Space

> In the case of a **Continuous Space**, the number of possible actions is **infinito**.

For example, in a *Self Driving Car agent* has an infinite number of possible actions since it can turn left 20°, 21,1°, 21,2°, honk, turn right 20°…

![img](images/action-space-02.png)










---

<div id="types-of-tasks"></div>

## Type of tasks (Episodic Task vs. Continuing Tasks)

> A *Task* is an **instance** of a *Reinforcement Learning problem*.

We can have two types of tasks:

 - **Episodic Task.**
 - **Continuous Task.**

Briefly, we can think of as:

![img](images/types-of-tasks-01.png)










---

<div id="exploitation-exploration-trade-off"></div>

## The Exploitation/Exploration trade-off

### Exploitation

> **The technique of "Exploitation" always tries the Action that it already knows has worked the most (or better).**

In summary, suppose a child is crying, the technique of **Exploitation** will always look for an Action that it has already tried and seen that it worked.

![img](images/exploitation-exploration-01.jpg)

Now let's analyze the following situation:

- **1st -** Suppose we were using a technique **(Action)** where the child only stopped crying after 5 minutes of rocking her;
- **2nd -** We discovered another technique **(Action)** where the child stopped crying instantly - For example, calling the mother.

It's obvious that the second option now gives a better **reward**. Therefore, the **Exploitation** technique will use the second option as the standard.

**NOTE:**  
The problem is that this approach will not lead us to an **optimal point**. This is because it always picks the best **Action** it already knows and ignores others that could yield better results **(For example, breastfeeding the child)**.

### Exploration

On the other hand, **Exploration** does the *exploration*. That is, it explores all possible **Actions**, calculating the average and percentage of all of them until finding the one that gives the best **reward**.

> **NOTE:**  
> The problem with **Exploration** is that depending on the problem, this can cost a lot of time and computational resources.

### Exploitation vs. Exploration

Well, the ideal is that when applying these concepts in practice, we carefully consider how much we will allocate for **Exploitation** and how much we will allocate for **Exploration**.




































































































<!--- ( Settings ) --->

---

<div id="settings"></div>

## Settings

**CREATE VIRTUAL ENVIRONMENT:**  
```bash
python -m venv ai-environment
```

**ACTIVATE THE VIRTUAL ENVIRONMENT (LINUX):**  
```bash
source ai-environment/bin/activate
```

**ACTIVATE THE VIRTUAL ENVIRONMENT (WINDOWS):**  
```bash
source ai-environment/Scripts/activate
```

**UPDATE PIP:**
```bash
python -m pip install --upgrade pip
```

**INSTALL PYTHON DEPENDENCIES:**  
```bash
pip install -U -v --require-virtualenv -r requirements.txt
```

**Now, Be Happy!!!** 😬





<!--- ( References ) --->

---

<div id="ref"></div>

## References

 - **General:**
   - [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
   - [Didatica Tech](https://didatica.tech/)

---

**R**odrigo **L**eite da **S**ilva - **drigols**
