"""
Prompt suite for thesis benchmarking.
Each prompt has: id, text, n_predict, input_category, output_category
"""

PROMPTS = [
    {
        "id": "P01",
        "name": "Factual Recall",
        "input_category": "short",
        "output_category": "short",
        "n_predict": 50,
        "text": ("What is edge computing? Answer in exactly two sentences."),
    },
    # {
    #     "id": "P02",
    #     "name": "Explanation",
    #     "input_category": "short",
    #     "output_category": "medium",
    #     "n_predict": 225,
    #     "text": (
    #         "Explain how tensor parallelism works in distributed machine learning "
    #         "inference. Write approximately one paragraph of five to seven sentences."
    #     ),
    # },
    {
        "id": "P02",
        "name": "Comparative Analysis",
        "input_category": "short",
        "output_category": "long",
        "n_predict": 525,
        "text": (
            "Compare pipeline parallelism and tensor parallelism for distributed "
            "inference on resource-constrained edge devices. Discuss the trade-offs "
            "in terms of communication overhead, latency, and scalability. Write "
            "approximately three to four paragraphs."
        ),
    },
    # {
    #     "id": "P04",
    #     "name": "Comprehension",
    #     "input_category": "medium",
    #     "output_category": "short",
    #     "n_predict": 50,
    #     "text": (
    #         "The Internet of Things (IoT) connects physical and digital devices using "
    #         "various communication protocols. IoT has three main characteristics: "
    #         "comprehensive perception, which uses sensors and RFID to collect information "
    #         "about objects at any time and place; reliable transmission, which sends "
    #         "data correctly using telecommunications networks; and intelligent "
    #         "processing, which uses computing technologies to process large volumes of "
    #         "data. However, IoT devices are often resource-constrained with limited "
    #         "computational abilities, memory, and power supply, making data processing "
    #         "challenging. Based on the paragraph above, what is the main challenge of "
    #         "IoT devices? Answer in exactly one sentence."
    #     ),
    # },
    # {
    #     "id": "P05",
    #     "name": "Summarization",
    #     "input_category": "medium",
    #     "output_category": "medium",
    #     "n_predict": 150,
    #     "text": (
    #         "Edge computing is a computing paradigm that performs computing and "
    #         "provides services at the network edge. The whole idea is to migrate "
    #         "the cloud's capabilities closer to where data is generated. Before the "
    #         "advent of edge computing, all data was transferred to a centralized cloud "
    #         "computing facility via the network for computation, storage, and analysis. "
    #         "However, with the increase in devices connected to the Internet of Things, "
    #         "the transmission rate and network bandwidth have become bottlenecks. "
    #         "Additionally, as most IoT devices generate personal and sensitive data, "
    #         "sending all the data to a remote server raises privacy concerns. The "
    #         "advent of edge computing will not replace cloud computing; instead, the "
    #         "two co-exist to complement each other's limitations. Summarize the above "
    #         "passage in approximately four to five sentences, focusing on why edge "
    #         "computing emerged and its relationship with cloud computing."
    #     ),
    # },
    # {
    #     "id": "P06",
    #     "name": "Reasoning + Elaboration",
    #     "input_category": "medium",
    #     "output_category": "long",
    #     "n_predict": 525,
    #     "text": (
    #         "A 2-node Raspberry Pi 5 cluster has 32 GB total RAM. The operating system "
    #         "on each node uses 1 GB. Each inference runtime instance uses 0.5 GB. The "
    #         "network communication buffer on each node requires 0.2 GB. A GGUF model in "
    #         "Q4_0 format has approximately 0.5 GB per billion parameters. Calculate the "
    #         "maximum model size in billions of parameters that this cluster can run using "
    #         "distributed inference. Show all steps of your reasoning. Then discuss "
    #         "what practical factors might reduce the actual usable memory below this "
    #         "theoretical maximum. Write approximately three to four paragraphs."
    #     ),
    # },
    {
        "id": "P03",
        "name": "Classification",
        "input_category": "long",
        "output_category": "short",
        "n_predict": 50,
        "text": (
            "Collaborative inference between small language models on edge and large "
            "language models on cloud infrastructure has emerged as a compelling approach "
            "for enabling intelligent services across resource-constrained environments. "
            "Instead of relying exclusively on centralized cloud inference, which "
            "introduces latency, privacy risks, and poor adaptation to local contexts, "
            "or compressing large models to fit on individual edge devices, which "
            "introduces loss in accuracy, collaborative inference exploits the combined "
            "strength of both approaches. The cloud-hosted LLMs provide strong "
            "generalization and zero-shot transfer across tasks, while the edge-hosted "
            "SLMs provide fast response times, data locality, and personalization. The "
            "intrinsic difference between the hardware foundations enables hierarchical "
            "task allocation. For example, SLMs can execute real-time perception, "
            "privacy-sensitive preprocessing, and intent filtering, and then offload "
            "complex generation and reasoning to LLMs only when necessary. For commodity "
            "IoT hardware such as Raspberry Pi clusters, the collaborative inference "
            "provides a practical balance. Individual edge devices lack the memory and "
            "computational power to execute large models on their own. However, when "
            "combined, the resources can be sufficient. Classify the approach described "
            "above into one of these categories: (a) single-device edge inference, "
            "(b) cloud-only inference, (c) edge-cloud collaborative inference, or (d) "
            "distributed edge-only inference. State only the letter and category name in "
            "one sentence."
        ),
    },
    # {
    #     "id": "P08",
    #     "name": "Critical Analysis",
    #     "input_category": "long",
    #     "output_category": "medium",
    #     "n_predict": 225,
    #     "text": (
    #         "Task assignment is the process of deciding upfront whether an edge SLM "
    #         "or a cloud LLM should handle a specific request. To achieve the right "
    #         "balance between saving energy, reducing lag, and keeping quality high, "
    #         "researchers usually use tools like lightweight scorers or bandit-based "
    #         "controllers to make routing choices on the fly. Some approaches use smaller "
    #         "models for easy tasks and larger ones only when needed. Tasks can be "
    #         "categorized by measuring cost effectiveness. Another scalable approach "
    #         "is Mixture-of-Experts, where systems route queries to specific cloud experts "
    #         "through a lightweight local gatekeeper. Agent-based methods extend this "
    #         "further by using planning agents to break down complex instructions and "
    #         "delegate subtasks to specialized agents. Task division allows Small "
    #         "Language Models and Large Language Models to collaborate on complementary "
    #         "subtasks by breaking down modular or hierarchical tasks into smaller "
    #         "components. This approach relies on three main strategies: routing, "
    #         "computation offloading, and early exit. Routing and forwarding techniques "
    #         "choose the most appropriate model on the fly during inference. Computation "
    #         "offloading distributes the inference workload between edge devices and cloud "
    #         "servers depending on live runtime conditions. Early-exit mechanisms give "
    #         "the system the ability to stop processing at middle layers if the model "
    #         "is confident enough. Critically evaluate which of these collaborative "
    #         "inference strategies would be most appropriate for a Raspberry Pi 5 cluster "
    #         "running IoT sensor data processing. Justify your answer in approximately "
    #         "five to seven sentences."
    #     ),
    # },
    {
        "id": "P04",
        "name": "Comprehensive Summarisation",
        "input_category": "long",
        "output_category": "long",
        "n_predict": 450,
        "text": (
            "Deploying LLMs on edge devices presents numerous challenges due to their "
            "resource-constrained nature. Edge devices are limited by computational "
            "memory resources, which prevents them from storing and executing LLMs "
            "directly. For example, GPT-3 contains billions of parameters and requires "
            "high-performance hardware such as GPUs or TPUs, which are mostly absent "
            "in edge devices. The gap between the rapid increase in computational "
            "complexity of LLMs and the slow growth in edge device capabilities is "
            "widening yearly. LLMs are characterized by substantial energy consumption "
            "during training and inference. Significant computational power is required "
            "to process large amounts of data and execute complex tasks, which contrasts "
            "with the strict energy constraints of edge environments. Even though edge "
            "computing reduces data privacy risks by processing data locally, it still "
            "requires rigorous security measures including robust encryption. Also, "
            "since edge environments consist of heterogeneous devices, standardizing "
            "protocols across devices is challenging. To tackle deployment issues, one "
            "approach is to compact the architecture to produce Small Language Models "
            "that operate in constrained environments. Compared to LLMs with hundreds of "
            "billions of parameters, SLMs only have millions to billions of parameters. "
            "Quantization is the most widely adopted technique, converting high-precision "
            "floating-point values into lower-precision formats. Knowledge Distillation "
            "transfers reasoning capabilities from large teacher models into smaller "
            "student models. Pruning removes redundant weights and connections that "
            "contribute little to the overall output. Write a comprehensive summary "
            "of the challenges and solutions described above. Organise your response "
            "into two sections: first the challenges, then the solutions. Write "
            "approximately three to four paragraphs total."
        ),
    },
]
