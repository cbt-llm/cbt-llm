from neo4j import GraphDatabase
from retrieve_snomed import retrieve_snomed_matches
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    # query_text = "I am not able to sleep at night, I am worried about my exams."
    # query_text = "I am overwhelmed with work/school."
    # query_text = "rumination"
    # query_text ="First off I would like to thank you for taking the time out to help me. But the problem is I’m depressed but nobody knows it. Half the reason I am is because I have no really close friends to hang out with or etc. While everybody is usually going to the movies, the beach, or somewhere fun I’m at home. My mother has started to notice it, she always suggests I hangout with my friends but truth is I don’t have the heart to tell her I don’t really have any. It started at the age of 11 when I started to notice I didn’t have a lot of friends like all the other kids did."
    # query_text = "I need strategies for managing my emotions."
    # query_text = "I am happy, this converstation really helped me."
    # query_text = "	From a teen in the U.S.: I’m 16 and this year was the worst for me. My parents got divorced, but I knew there was no bad blood between each other and me and them. Yet, it still sucked knowing theyre divorced now. I understand why it happened and I know it’s a necessary evil. However, it affected my school work so bad."

    query_text = " "

    results = retrieve_snomed_matches(driver, query_text, k=5)

    print("\nTop k SNOMED matches:")
    for r in results:
        if r['term'] == "No matches found":
            print(r['term'])
        else:
            print(f"- {r['term']}  (code: {r['code']}, score: {r['score']:.4f})")
            if r['relations']:
                for rel in r['relations']:
                    print(f"    -> {rel['type']} -> {rel['target']}")

    driver.close()

if __name__ == "__main__":
    main()
 