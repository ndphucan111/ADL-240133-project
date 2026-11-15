import csv
import os
import sys
import time

if os.getenv("ENGINE", "v2") == "baseline":
    from run_pdf import build_rag, ask  # Task 1 code, the baseline

    def config_summary():
        return "ENGINE=baseline (run_pdf.py: llm=qwen2.5:0.5b chunk=1000 top_k=5)"
else:
    from rag_v2 import build_rag, ask, config_summary

PDF = "major_handbook_pdfs/CS_MajorHandbook.pdf"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

# (question, expected answer, pages where the answer is; (0,) = not in the handbook)
QUESTIONS = [
    # ---- original 20 questions ----
    ("What is the major code of the B.S. Computer Science program?",
     "7480101", (1,)),
    ("What are the Program Educational Objectives (PEOs)?",
     "PEO1: skills for employment or graduate school; PEO2: ethics, collaboration, communication; PEO3: innovators who solve CS challenges", (4,)),
    ("What jobs have Computer Science graduates gotten?",
     "Software engineer, testing engineer, AI engineer, data engineer, business analyst, product owner, product development manager", (5,)),
    ("Which graduate programs have CS students applied to?",
     "MSc/PhD Computer Science, MSc Data Science, MSc Bioinformatics", (5, 6)),
    ("How many credits are the general curriculum requirements?",
     "40 credits and 10 noncredit items", (6,)),
    ("Which four core courses must students take?",
     "Global Humanities, Modern Vietnamese Culture and Society, Design and Systems Thinking, and one QUEST course", (6,)),
    ("How many credits are the major requirements?",
     "56 credits", (7,)),
    ("What must a student complete to formally declare Computer Science as a major?",
     "At least two Computer Science courses (MATH 202 also counts)", (7,)),
    ("How many courses and credits does the Computer Science minor require?",
     "Six courses, 24 credits", (7,)),
    ("What is the minimum major GPA to enroll in CS 401 Capstone I?",
     "3.5 and approval of a successful application", (9,)),
    ("What degree is awarded after completing this program?",
     "B.S. Computer Science (the handbook text wrongly says Applied Mathematics)", (6,)),
    ("How many foundation courses are required, and which are they?",
     "Five: Intro to Programming, Data Structures & Algorithms, Computer Organization, Discrete Math, Algorithms & Theory of Computing", (8, 10)),
    ("How many advanced courses are required?",
     "Four", (7, 8, 10)),
    ("How many courses are in the Computer Science minor according to the flowchart?",
     "7 courses (the text says 6, the handbook is inconsistent)", (10,)),
    ("What is the prerequisite of CS302 Algorithms & Theory of Computing?",
     "CS201 and MATH 202", (13, 17)),
    ("What is the prerequisite of CS311 Advanced Deep Learning?",
     "CS313", (13, 18)),
    ("Which programming language is mainly used in CS201?",
     "Java (some Python/C++ also introduced)", (14,)),
    ("Which course teaches finite automaton, pushdown automaton and Turing machine?",
     "CS302 Algorithms & Theory of Computing", (17,)),
    ("Which courses are currently on holding?",
     "CS210, CS303, CS306, CS310, CS3xx Bioinformatics", (13,)),
    ("How much is the tuition fee for the Computer Science major?",
     "Not in the handbook. A good system should say it does not know", (0,)),
    # ---- 5 NEW questions (not used while improving the system) ----
    ("What is the prerequisite of CS101 Introduction to Programming?",
     "None (CS104 is recommended)", (13, 14)),
    ("How often is CS104 Introduction to Computer Science offered?",
     "Once every two years", (13,)),
    ("Which course has CORE105 as its prerequisite?",
     "CS203 Computer Organization", (13, 14)),
    ("Which course covers Transfer Learning, GAN and Graph Neural Networks?",
     "CS311 Advanced Deep Learning", (18,)),
    ("How many credits is each course unless indicated otherwise?",
     "4 credits (one semester)", (13,)),
]

if os.getenv("ONLY_NEW") == "1":
    QUESTIONS = QUESTIONS[20:]

print("CONFIG: " + config_summary())
db, llm = build_rag(PDF)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["question", "expected", "expected_page", "answer", "top_pages",
                "hit", "seconds", "label", "reason"])
    for q, expected, pages_ok in QUESTIONS:
        t0 = time.time()
        try:
            r = ask(q, db, llm)
        except Exception as e:
            r = {"answer": "[ERROR: " + str(e)[:100] + "]", "sources": []}
        sec = round(time.time() - t0, 1)
        got = [s["page"] for s in r["sources"]]
        hit = "" if pages_ok == (0,) else int(any(p in got for p in pages_ok))
        w.writerow([q, expected, "/".join(str(p) for p in pages_ok),
                    r["answer"], got, hit, sec, "", ""])
        print("   -> " + r["answer"][:120].replace("\n", " "))
    w.writerow([])
    w.writerow(["CONFIG", config_summary()])
print("Saved to", OUT)