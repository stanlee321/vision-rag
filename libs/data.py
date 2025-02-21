template = (
    "Imagine you are a data scientist's assistant and "
    "you answer a recruiter's questions about the data scientist's experience."
    "Here is some context from the data scientist's "
    "resume related to the query::\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, "
    "Please respond to the following inquiry:\n\n"
    "Question: {query_str}\n\n"
    "Answer succinctly and ensure your response is "
    "clear to someone without a data science background."
    
    "The data scientist's name is Gobia."
)

response_mode_dict = {
    "REFINE": {
        "value": "refine",
        "description": (
            "Refine is an iterative way of generating a response. "
            "We first use the context in the first node, along with the query, to generate an initial answer. "
            "We then pass this answer, the query, and the context of the second node as input into a 'refine prompt' "
            "to generate a refined answer. We refine through N-1 nodes, where N is the total number of nodes."
        ),
    },
    "COMPACT": {
        "value": "compact",
        "description": (
            "Compact and refine mode first combine text chunks into larger consolidated chunks "
            "that more fully utilize the available context window, then refine answers across them. "
            "This mode is faster than refine since we make fewer calls to the LLM."
        ),
    },
    "SIMPLE_SUMMARIZE": {
        "value": "simple_summarize",
        "description": (
            "Merge all text chunks into one, and make an LLM call. "
            "This will fail if the merged text chunk exceeds the context window size."
        ),
    },
    "TREE_SUMMARIZE": {
        "value": "tree_summarize",
        "description": (
            "Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query. "
            "The tree is built in a bottom-up fashion, and in the end, the root node is returned as the response."
        ),
    },
    "GENERATION": {
        "value": "generation",
        "description": "Ignore context, just use LLM to generate a response.",
    },
    "NO_TEXT": {
        "value": "no_text",
        "description": "Return the retrieved context nodes, without synthesizing a final response.",
    },
    "CONTEXT_ONLY": {
        "value": "context_only",
        "description": "Returns a concatenated string of all text chunks.",
    },
    "ACCUMULATE": {
        "value": "accumulate",
        "description": (
            "Synthesize a response for each text chunk, and then return the concatenation."
        ),
    },
    "COMPACT_ACCUMULATE": {
        "value": "compact_accumulate",
        "description": (
            "Compact and accumulate mode first combine text chunks into larger consolidated chunks "
            "that more fully utilize the available context window, then accumulate answers for each of them "
            "and finally return the concatenation. "
            "This mode is faster than accumulate since we make fewer calls to the LLM."
        ),
    },
}