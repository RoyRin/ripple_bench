# Setting that we care about:
The field of unlearning has blossomed, but one of the problems in the field is the proliferation of different kinds, and sometimes how different ones are used. 

Through the literature and through our own reasoning, we extract 3 high-level definitions of unlearning:
1. Probe-Guardedness:   
    * LEACE-style, you may want this for fairness
2. CCU 
    * Data-selection + data-unlearning. You may want this for something like copyright and privacy.
3. Capability and/or Knowledge removal
    * This definition is still a work in progress, though significant insights are being made. Generally what we want is that on a particular task, a model behaves like a novice rather than an expert on some specific task.
    * Reasoning from how humans operate, one constructs at least 2 definitions: 
        * knowledge unlearning, where a model's ability to do factual recall or classification is novice-like,
        * capability removal, 

Having clear definitions of unlearning is vital for making progress, as it allows the field to progress for two reasons:
1. It allows researchers and engineers to align on a common goal, and produce technical insights into improvements on the goal. 
2. Being able to formally state the measure of success clearly, allows one to reason about what a perfect unlearner would do (for example, a perfect data-unlearner would be a model fully-retrained on the retain set).

Unfortunately, defining what "knowledge" is and what "capabilities" are is not only a technically challenging problem, it's a philosophically unresolved question. Many papers have taken different measures of success (ELM, TARS, RMU, etc.). We propose a means to make progress in this direction, through increased characterization of what an unlearning algorithm is doing -- that is, we develop a framework for evaluating what an unlearning algorithm does to a model, giving us insight into how it is doing it.


# The danger of dual-use
We are not the first to observe that knowledge and "capabilities" tend to be dual-use; 


# Insight into types of unlearning
We observe that there are different "ways" in which unlearning may work:
* one may unlearn the capability to create a bomb (a common safety task) by removing the facts necessary to create a bomb (e.g. **say fact about bombs here**). 
* One may also remove the model's ability to compose facts relating to a topic - e.g. given a capability, such as composing 2 facts together, one can remove a model's ability to apply that capability two facts relating to some specific topic.
* One can reduce the internal structure a model learned for examples relating to some topic, concept, or capability.

Importantly, that for each of these notions we observe that  

# Experiments that we want to run:
1. Given some topic, create a graph around them, with grounded questions - ripple effect

Question: can we find some way to predict what will be hurt, based on some unlearning
* need to read ELM paper more carefully, and see exactly what they are doing - they focus on unlearning only 1 topic


Other possible experiments:
1. create synthetic experiment 
