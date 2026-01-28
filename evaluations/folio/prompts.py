"""
Prompts and few-shot examples for FOLIO evaluation with dynamic CSD.

This module provides prompt templates and few-shot examples adapted from CRANE
for first-order logic (FOL) parsing tasks on the FOLIO dataset.
"""

from typing import Dict, List, Optional, Tuple
import re


# Grammar description for FOL
FOL_GRAMMAR_DESCRIPTION = """The grammar of the first-order logic formula is defined as follows:
1) logical conjunction of expr1 and expr2: expr1 {and} expr2
2) logical disjunction of expr1 and expr2: expr1 {or} expr2
3) logical exclusive disjunction of expr1 and expr2: expr1 {xor} expr2
4) logical negation of expr1: {not}expr1
5) expr1 implies expr2: expr1 {implies} expr2
6) expr1 if and only if expr2: expr1 {iff} expr2
7) logical universal quantification: {forall} x
8) logical existential quantification: {exists} x. These are the ONLY operations in the grammar."""


# Few-shot examples for FOLIO evaluation (matching CRANE paper format)
# Each example contains:
# - problem: The natural language premises
# - question: The statement to evaluate (true/false/uncertain)
# - predicates: List of predicate definitions
# - premises: FOL encoding of premises (NO delimiters - CRANE paper format)
# - conclusion: FOL encoding of the conclusion
# - answer: True/False/Uncertain

FOLIO_FEW_SHOT_EXAMPLES = [
    {
        "problem": """All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.""",
        "question": "Based on the above information, is the following statement true, false, or uncertain? Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.",
        "predicates": [
            ("Dependent(x)", "x is a person dependent on caffeine."),
            ("Drinks(x)", "x regularly drinks coffee."),
            ("Jokes(x)", "x jokes about being addicted to caffeine."),
            ("Unaware(x)", "x is unaware that caffeine is a drug."),
            ("Student(x)", "x is a student."),
        ],
        "premises": [
            ("{forall} x (Drinks(x) {implies} Dependent(x))", "All people who regularly drink coffee are dependent on caffeine."),
            ("{forall} x (Drinks(x) {xor} Jokes(x))", "People either regularly drink coffee or joke about being addicted to caffeine."),
            ("{forall} x (Jokes(x) {implies} {not}Unaware(x))", "No one who jokes about being addicted to caffeine is unaware that caffeine is a drug."),
            ("(Student(rina) {and} Unaware(rina)) {xor} {not}(Student(rina) {or} Unaware(rina))", "Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug."),
        ],
        "conclusion": ("Jokes(rina) {xor} Unaware(rina)", "Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug."),
        "answer": "True",
    },
    {
        "problem": """Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.""",
        "question": "Based on the above information, is the following statement true, false, or uncertain? Miroslav Venhoda loved music.",
        "predicates": [
            ("Czech(x)", "x is a Czech person."),
            ("ChoralConductor(x)", "x is a choral conductor."),
            ("Musician(x)", "x is a musician."),
            ("Love(x, y)", "x loves y."),
            ("Author(x, y)", "x is the author of y."),
            ("Book(x)", "x is a book."),
            ("Publish(x, y)", "x is published in year y."),
            ("Specialize(x, y)", "x specializes in y."),
        ],
        "premises": [
            ("Czech(miroslav) {and} ChoralConductor(miroslav) {and} Specialize(miroslav, renaissance) {and} Specialize(miroslav, baroque)", "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music."),
            ("{forall} x (ChoralConductor(x) {implies} Musician(x))", "Any choral conductor is a musician."),
            ("{exists} x (Musician(x) {and} Love(x, music))", "Some musicians love music."),
            ("Book(methodOfStudyingGregorianChant) {and} Author(miroslav, methodOfStudyingGregorianChant) {and} Publish(methodOfStudyingGregorianChant, year1946)", "Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant."),
        ],
        "conclusion": ("Love(miroslav, music)", "Miroslav Venhoda loved music."),
        "answer": "Uncertain",
    },
    {
        "problem": """People in this club who perform in school talent shows often attend and are very engaged with school events. People in this club either perform in school talent shows often or are inactive and disinterested community members. People in this club who chaperone high school dances are not students who attend the school. All people in this club who are inactive and disinterested members of their community chaperone high school dances. All young children and teenagers in this club who wish to further their academic careers and educational opportunities are students who attend the school. Bonnie is in this club and she either both attends and is very engaged with school events and is a student who attends the school or is not someone who both attends and is very engaged with school events and is not a student who attends the school.""",
        "question": "Based on the above information, is the following statement true, false, or uncertain? Bonnie performs in school talent shows often.",
        "predicates": [
            ("InClub(x)", "x is a member of the club."),
            ("Perform(x)", "x performs in school talent shows."),
            ("Attend(x)", "x attends school events."),
            ("Engaged(x)", "x is very engaged with school events."),
            ("Inactive(x)", "x is an inactive and disinterested community member."),
            ("Chaperone(x)", "x chaperones high school dances."),
            ("Student(x)", "x is a student who attends the school."),
            ("Wish(x)", "x wishes to further their academic careers and educational opportunities."),
            ("Young(x)", "x is a young child."),
            ("Teenager(x)", "x is a teenager."),
        ],
        "premises": [
            ("{forall} x (InClub(x) {and} Perform(x) {implies} Attend(x) {and} Engaged(x))", "People in this club who perform in school talent shows often attend and are very engaged with school events."),
            ("{forall} x (InClub(x) {implies} (Perform(x) {xor} Inactive(x)))", "People in this club either perform in school talent shows often or are inactive and disinterested community members."),
            ("{forall} x (InClub(x) {and} Chaperone(x) {implies} {not}Student(x))", "People in this club who chaperone high school dances are not students who attend the school."),
            ("{forall} x (InClub(x) {and} Inactive(x) {implies} Chaperone(x))", "All people in this club who are inactive and disinterested members of their community chaperone high school dances."),
            ("{forall} x (InClub(x) {and} (Young(x) {or} Teenager(x)) {and} Wish(x) {implies} Student(x))", "All young children and teenagers in this club who wish to further their academic careers and educational opportunities are students who attend the school."),
            ("InClub(bonnie) {and} ((Attend(bonnie) {and} Engaged(bonnie) {and} Student(bonnie)) {xor} ({not}(Attend(bonnie) {and} Engaged(bonnie)) {and} {not}Student(bonnie)))", "Bonnie is in this club and she either both attends and is very engaged with school events and is a student who attends the school or is not someone who both attends and is very engaged with school events and is not a student who attends the school."),
        ],
        "conclusion": ("InClub(bonnie) {and} Perform(bonnie)", "Bonnie performs in school talent shows often."),
        "answer": "Uncertain",
    },
]


def extract_predicates_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract predicate definitions from problem text.
    
    This is a heuristic approach - in practice, the model generates predicates.
    Returns a list of (predicate_signature, description) tuples.
    """
    # This would typically be done by the model during generation
    # For now, return an empty list as predicates are defined during generation
    return []


def extract_constants_from_problem(problem: str, question: str) -> List[str]:
    """
    Extract potential constants (proper nouns, specific entities) from the problem.
    
    These are typically lowercase identifiers in FOL that represent specific entities.
    """
    # Find capitalized words that are likely proper nouns
    combined_text = problem + " " + question
    
    # Pattern for proper nouns (capitalized words not at start of sentence)
    words = combined_text.split()
    proper_nouns = set()
    
    for i, word in enumerate(words):
        # Clean the word
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if clean_word and clean_word[0].isupper():
            # Skip words at start of sentences and common words
            if i > 0 and not words[i-1].endswith('.'):
                # Convert to lowercase for constant name
                constant = clean_word.lower()
                if len(constant) > 1:  # Skip single letters
                    proper_nouns.add(constant)
    
    return list(proper_nouns)


def format_predicates_section(predicates: List[Tuple[str, str]]) -> str:
    """Format the predicates section of the FOL solution."""
    lines = ["Predicates:"]
    for pred_sig, description in predicates:
        lines.append(f"{pred_sig} ::: {description}")
    return "\n".join(lines)


def format_premises_section(premises: List[Tuple[str, str]]) -> str:
    """Format the premises section with constrained windows."""
    lines = ["Premises:"]
    for fol_expr, nl_description in premises:
        lines.append(f"{fol_expr} ::: {nl_description}")
    return "\n".join(lines)


def format_conclusion_section(conclusion: Tuple[str, str]) -> str:
    """Format the conclusion section with constrained window."""
    fol_expr, nl_description = conclusion
    return f"Conclusion:\n{fol_expr} ::: {nl_description}"


def format_example_solution(example: Dict) -> str:
    """
    Format a complete example solution for few-shot prompting.

    Matches the CRANE paper format (no << >> delimiters in examples).
    """
    # Get the question text for the introduction
    question_text = example["question"]
    # Extract just the statement part after "true, false, or uncertain?"
    if "uncertain?" in question_text:
        statement = question_text.split("uncertain?")[-1].strip()
    else:
        statement = example.get("conclusion", ("", ""))[1]

    # Format the predicates
    predicates_section = format_predicates_section(example["predicates"])

    # Format the premises
    premises_section = format_premises_section(example["premises"])

    # Format the conclusion
    conclusion_section = format_conclusion_section(example["conclusion"])

    # Combine all sections with CRANE-style introduction
    # Include Answer at the end so the model learns to output it
    answer = example.get("answer", "")
    solution = f"""We take three steps: first, we define the necessary predicates and premises, and finally, we encode the question '{statement}' in the conclusion. Now, we will write only the logic program, nothing else.
{predicates_section}
{premises_section}
{conclusion_section}
Answer: {answer}"""

    return solution


def format_example(example: Dict) -> str:
    """Format a complete few-shot example (problem + solution) in CRANE paper format."""
    return f"""Problem:
{example["problem"]}
Question:
{example["question"]}
###

{format_example_solution(example)}"""


def make_folio_prompt(
    problem: str,
    question: str,
    num_examples: int = 2,
    include_grammar_description: bool = True,
) -> str:
    """
    Create a prompt for FOLIO evaluation matching the CRANE paper format.

    Args:
        problem: The natural language premises
        question: The statement to evaluate
        num_examples: Number of few-shot examples to include
        include_grammar_description: Whether to include the FOL grammar description

    Returns:
        The formatted prompt string
    """
    # Start with the system instruction (CRANE paper format)
    parts = [
        "Given a problem description and a question. The task is to parse the problem and the question into first-order logic formulas."
    ]

    # Add grammar description if requested
    if include_grammar_description:
        parts.append(FOL_GRAMMAR_DESCRIPTION)

    parts.append("------")
    parts.append("")
    parts.append("Answer the question EXACTLY like the examples.")
    parts.append("")

    # Add few-shot examples
    examples_to_use = FOLIO_FEW_SHOT_EXAMPLES[:num_examples]
    for example in examples_to_use:
        parts.append(format_example(example))
        parts.append("------")
        parts.append("")

    # Add the actual problem (CRANE paper format)
    parts.append(f"""Problem:
{problem}
Question:
{question}
###
""")

    return "\n".join(parts)


def make_folio_prompt_no_cot(
    problem: str,
    question: str,
    num_examples: int = 2,
) -> str:
    """
    Create a prompt for FOLIO evaluation without chain-of-thought.
    
    This version asks directly for the answer without FOL formalization.
    """
    parts = [
        "Given a problem description and a question. Determine if the statement in the question is true, false, or uncertain based on the premises.",
        "",
    ]
    
    # Add few-shot examples (simplified)
    examples_to_use = FOLIO_FEW_SHOT_EXAMPLES[:num_examples]
    for example in examples_to_use:
        parts.append(f"""Problem:
{example["problem"]}
Question:
{example["question"]}

Answer: {example["answer"]}
------
""")
    
    # Add the actual problem
    parts.append(f"""Problem:
{problem}
Question:
{question}

Answer:""")
    
    return "\n".join(parts)


# Constants used for window detection during generation
CONSTRAINT_START = "<<"
CONSTRAINT_END = ">>"
ANSWER_PREFIX = "Answer:"
FOL_SOLUTION_MARKER = "FOL Solution:"
PREDICATES_MARKER = "Predicates:"
PREMISES_MARKER = "Premises:"
CONCLUSION_MARKER = "Conclusion:"
