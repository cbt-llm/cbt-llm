import csv
from neo4j import GraphDatabase
from cbt_llm.retrieve_snomed import retrieve_snomed_matches
from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


EVAL_DATA = [
    (
        "I’ve been feeling constantly overwhelmed and on edge since everything changed so suddenly, and even small daily tasks now feel hard to cope with.",
        "Stress and adjustment reaction"
    ),
    (
        "I feel completely drained and worn down, as if I have nothing left to give, and even getting through the day feels exhausting.",
        "Physical and emotional exhaustion state"
    ),
    (
        "I feel intense fear and uneasiness when I’m away from the people I’m close to, and it’s hard for me to relax or focus until I know they’re okay.",
        "Separation anxiety"
    ),
    (
        "I’m struggling to adapt to recent changes in my life, and I feel unsettled and unsure of myself in ways I didn’t expect.",
        "Adjustment reaction of adult life"
    ),
    (
        "I keep going over the same thoughts again and again, unable to let them go, even when I know it’s not helping me feel better.",
        "Rumination disorder"
    ),
    (
        "I’ve been feeling persistently low and weighed down since the changes in my life, and the sadness hasn’t lifted even as time has passed.",
        "Prolonged depressive adjustment reaction"
    ),
    (
        "I’ve been feeling deeply low for a long time now, with little motivation or hope, and it’s hard to remember what it feels like to be myself again.",
        "Chronic bipolar I disorder, most recent episode depressed"
    ),
    (
        "I feel blocked and stuck when it comes to my studies or work, even though I want to move forward, and it leaves me feeling frustrated and discouraged.",
        "Specific academic or work inhibition"
    ),
    (
        "I feel like I’m not performing to my potential academically, even though I put in effort, and it leaves me feeling disappointed and unsure of my abilities.",
        "Academic underachievement disorder"
    ),
    (
        "I feel constantly overwhelmed by the responsibility of caring for someone else, and it’s becoming harder to manage my own needs without feeling guilty or exhausted.",
        "Career stress syndrome"
    ),
    (
        "I’ve been feeling down nearly every day for weeks, with little interest in anything, and it’s starting to affect my ability to function.",
        "Depressive disorder"
    ),
    (
        "This is the first time I’ve ever felt this persistently hopeless and exhausted for weeks, and it’s making it hard to work, eat, or enjoy anything.",
        "Major depression, single episode"
    ),
    (
        "I struggle to connect with other kids/people my age, I don’t really know how to join conversations, and social situations feel confusing and isolating.",
        "Childhood or adolescent disorder of social functioning"
    ),
    (
        "I keep having vivid nightmares about what happened, and I wake up terrified and drenched in sweat, like I’m back there again.",
        "Nightmares associated with chronic post-traumatic stress disorder"
    ),
    (
        "I feel panicky when I’m away from the people I’m closest to, and I can’t relax until I know they’re safe and okay.",
        "Separation anxiety"
    ),
    (
        "After drinking, I get intense anxiety and restlessness that feels out of proportion, and it can last into the next day.",
        "Alcohol-induced anxiety disorder"
    ),
    (
        "Ever since coming back, I feel constantly on edge and drained, like my body and mind are stuck in survival mode and I can’t reset.",
        "Combat fatigue"
    ),
    (
        "I can’t stop fixating on how flawed my appearance looks, and I spend hours checking or trying to hide what I see as defects.",
        "Body dysmorphic disorder"
    ),
    (
        "My partner and I keep running into the same painful sexual difficulties, and it’s creating distress and tension between us.",
        "Sexual relationship disorder"
    ),
    (
        "Reading takes me way longer than it should, I mix up words or lose my place often, and it’s been a persistent issue since school.",
        "Developmental reading disorder"
    ),
    (
        "I’ve been really disoriented lately—sometimes I don’t know where I am for a moment, and my thinking feels cloudy and confused.",
        "Confusional state"
    ),
    (
        "After giving birth, my mental health has gotten much worse, and it’s making recovery and caring for the baby feel overwhelming.",
        "Mental disorder in mother complicating childbirth"
    ),
    (
        "I can’t stay focused at work, I procrastinate even on simple tasks, and my mind feels like it’s constantly jumping around.",
        "Adult attention deficit hyperactivity disorder"
    ),
    (
        "I’m barely sleeping, and it feels tied to stress and anxious thoughts—my mind won’t shut off even when I’m exhausted.",
        "Hyposomnia co-occurrent and due to psychological disorder"
    ),
    (
        "I act on urges without thinking, and even when I know it will cause problems, it feels almost impossible to stop myself in the moment.",
        "Impulse control disorder"
    ),
    (
        "I find myself drinking huge amounts of water even when I’m not thirsty, and it feels driven by anxiety more than anything physical.",
        "Psychogenic polydipsia"
    ),
    (
        "I keep pulling out my hair without meaning to, especially when I’m stressed, and then I feel ashamed when I notice bald spots.",
        "Trichotillomania"
    ),
    (
        "I keep gambling even when I promise myself I’ll stop, and I’m chasing losses even though it’s hurting my finances and relationships.",
        "Compulsive gambling"
    ),
    (
        "I sometimes get an intense urge to start fires, and the impulse feels exciting and relieving in a way that scares me afterward.",
        "Pyromania"
    ),
    (
        "I keep taking things I don’t need and could easily afford, and the urge builds up until I do it, then I feel guilty afterward.",
        "Kleptomania"
    ),
    (
        "I attend classes, get my work done and get good grades. I laugh with friends and family, pretending everything is going well. But, when I am alone, I feel the heaviness of it all, like nothing matters.",
        "Masked depression"
    ),
    (
        "My mood swings between feeling very low and unusually energized, and it’s starting to interfere with my daily life.",
        "Mood disorder"
    ),
    (
        "From a young age, I struggled emotionally and behaviorally in ways that made school and friendships very difficult.",
        "Mental disorder in childhood"
    ),
    (
        "Growing up without emotional support has left me feeling empty, disconnected, and unsure how to form close bonds.",
        "Emotional deprivation syndrome"
    ),
    (
        "I often break rules, argue with authority figures, and have trouble controlling my behavior at home and school.",
        "Disruptive behavior disorder"
    ),
    (
        "I act aggressively toward others and don’t really form close friendships, preferring to stay isolated and hostile.",
        "Aggressive type unsocialized behavior disorder"
    ),
    (
        "I have trouble focusing on tasks, miss details, and get distracted easily, even when I try hard to pay attention.",
        "Attention deficit hyperactivity disorder, predominantly inattentive type"
    ),
    (
        "I’ve repeatedly broken rules, gotten into fights, and ignored social norms, even when I knew there would be consequences.",
        "Conduct disorder"
    ),
    (
        "I act out and get into trouble a lot, but underneath it I also feel persistently sad and hopeless.",
        "Depressive conduct disorder"
    ),
    (
        "Relationships during my teen years were full of conflict, misunderstandings, and emotional strain that I couldn’t handle.",
        "Childhood and adolescent relationship problem"
    ),
    (
        "I felt intense jealousy toward my sibling growing up, to the point where it affected my behavior and emotions.",
        "Sibling jealousy"
    ),
    (
        "I showed serious rule-breaking and aggressive behavior while remaining socially isolated from peers.",
        "Conduct disorder - unsocialized"
    ),
    (
        "I behaved aggressively and rejected social norms, without forming meaningful peer connections.",
        "Aggressive unsocial conduct disorder"
    ),
    (
        "I broke rules and acted out but wasn’t physically aggressive, and I didn’t really connect with peers either.",
        "Nonaggressive unsocial conduct disorder"
    ),
    (
        "I frequently argue with authority figures, refuse to comply with rules, and feel easily annoyed or resentful.",
        "Oppositional defiant disorder"
    ),
    (
        "I engaged in problematic behaviors mainly in group settings, often influenced by peers.",
        "Socialized behavior disorder"
    ),
    (
        "Serious rule-breaking behaviors started during my teenage years rather than early childhood.",
        "Conduct disorder, adolescent-onset type"
    ),
    (
        "I showed persistent rule-breaking behavior without aggression and struggled to connect socially.",
        "Unaggressive type unsocialized behavior disorder"
    ),
    (
        "Reading has always been unusually difficult for me despite normal intelligence and effort.",
        "Developmental reading disorder"
    ),
    (
        "I struggle specifically with recognizing and understanding written words, even though other skills are fine.",
        "Specific reading disorder"
    ),
    (
        "After being in intensive care and receiving medications, I became severely confused and disconnected from reality.",
        "Drug-induced intensive care psychosis"
    ),
    (
        "I’ve been persistently confused and disoriented for a long time, with memory and awareness problems.",
        "Chronic confusional state"
    ),
    (
        "I feel low and unmotivated most days, but not to the extreme level of major depression.",
        "Minor depressive disorder"
    ),
    (
        "Before my period, I experience severe mood swings, irritability, and emotional distress that disrupt daily life.",
        "Premenstrual dysphoric disorder"
    ),
    (
        "After childbirth, I felt persistently low and overwhelmed, though still able to function somewhat.",
        "Mild postnatal depression"
    ),
    (
        "After childbirth, I experienced intense depression, difficulty bonding, and trouble coping day to day.",
        "Severe postnatal depression"
    ),
    (
        "Shortly after giving birth, I experienced sudden mood swings, tearfulness, and emotional sensitivity.",
        "Maternity blues"
    ),
    (
        "I’ve felt mildly depressed for years, with low energy and self-esteem but no clear episodes.",
        "Dysthymia"
    ),
    (
        "I worry excessively about many things almost every day, and it’s hard to control the anxiety.",
        "Generalized anxiety disorder"
    ),
    (
        "I have sudden panic attacks and avoid places where escape might be difficult, like crowded areas or public transport.",
        "Panic disorder with agoraphobia"
    ),
    (
        "I’ve been feeling low for weeks and nothing really lifts my mood anymore.",
        "Depressive disorder"
    ),
    (
        "This is the first time I’ve felt this deeply depressed and unable to function for so long.",
        "Major depression, single episode"
    ),
    (
        "I keep having terrifying nightmares about the trauma, and I wake up panicking every night.",
        "Nightmares associated with chronic post-traumatic stress disorder"
    ),
    (
        "I get extremely anxious whenever I’m away from my partner or family.",
        "Separation anxiety"
    ),
    (
        "I can’t relax when the people I’m close to aren’t around; I feel on edge the whole time.",
        "Separation anxiety"
    ),
    (
        "After drinking, my anxiety spikes badly and I feel shaky and panicked the next day.",
        "Alcohol-induced anxiety disorder"
    ),
    (
        "Sometimes I feel suddenly disoriented and confused, like I don’t know where I am for a moment.",
        "Confusional state"
    ),
    (
        "I can’t stay focused on tasks at work and my mind keeps jumping from one thing to another.",
        "Adult attention deficit hyperactivity disorder"
    ),
    (
        "I keep gambling even when I know it’s hurting my finances and relationships.",
        "Compulsive gambling"
    ),
    (
        "I sometimes steal small things I don’t even need and then feel guilty afterward.",
        "Kleptomania"
    ),
    (
        "Everyone thinks I’m fine because I’m functioning, but inside I feel empty and miserable.",
        "Masked depression"
    ),
    (
        "I smile and act normal around people, but when I’m alone the sadness hits hard.",
        "Masked depression"
    ),
    (
        "My mood swings from really low to unusually energized, and it’s hard to predict how I’ll feel.",
        "Mood disorder"
    ),
    (
        "I get into trouble a lot for breaking rules and ignoring authority, even when I know I shouldn’t.",
        "Conduct disorder"
    ),
    (
        "I act out and get aggressive, but I also feel deeply sad underneath it all.",
        "Depressive conduct disorder"
    ),
    (
        "Relationships during my teen years were full of conflict and misunderstandings I couldn’t handle.",
        "Childhood and adolescent relationship problem"
    ),
    (
        "I used to feel intense jealousy toward my sibling that I couldn’t control.",
        "Sibling jealousy"
    ),
    (
        "I feel persistently low and unmotivated, but not completely unable to function.",
        "Minor depressive disorder"
    ),
    (
        "After giving birth, I had sudden mood swings and cried for no clear reason.",
        "Maternity blues"
    ),
    (
        "I worry constantly about everything, even small things, and can’t switch it off.",
        "Generalized anxiety disorder"
    ),
]

EMBEDDING_MODES = ["mpnet", "sapbert", "bioreddit", "mentalbert"]
K_VALUES = [1, 3, 5]



import re
import unicodedata

def norm(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    # normalize apostrophes/quotes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # remove semantic tags
    for suffix in [" (disorder)", " (finding)", " (situation)", " (event)", " (procedure)"]:
        text = text.replace(suffix, "")

    # remove punctuation that frequently differs in SNOMED terms
    text = re.sub(r"[,\.;:]", "", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def recall_at_k(retrieved_terms, target, k):
    return int(norm(target) in [norm(t) for t in retrieved_terms[:k]])


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    
    recall_csv = open("recall_summary.csv", "w", newline="", encoding="utf-8")
    recall_writer = csv.writer(recall_csv)
    recall_writer.writerow(["Embedding", "Recall@1", "Recall@3", "Recall@5"])

    qual_csv = open("qualitative_examples.csv", "w", newline="", encoding="utf-8")
    qual_writer = csv.writer(qual_csv)
    qual_writer.writerow([
        "Embedding",
        "Query",
        "Ground Truth Disorder",
        "Rank",
        "Retrieved Concept",
        "Is Correct"
    ])

    print("\n==============================")
    print(" SNOMED RETRIEVAL EVALUATION ")
    print("==============================")

    for mode in EMBEDDING_MODES:
        hits = {k: 0 for k in K_VALUES}

        print(f"\nEmbedding Model: {mode.upper()}")
        print("--------------------------------------------")

        for query, target in EVAL_DATA:
            results = retrieve_snomed_matches(
                driver, query, mode=mode, k=max(K_VALUES)
            )

            retrieved_terms = [
                r["term"] for r in results if r["term"] != "No matches found"
            ]

            print(f"\nQuery: {query}")
            print(f"Ground Truth: {target}")
            print("Retrieved:")

            for rank, term in enumerate(retrieved_terms, start=1):
                correct = norm(term) == norm(target)
                marker = "" if correct else ""
                print(f"  {rank}. {term} {marker}")

                
                qual_writer.writerow([
                    mode,
                    query,
                    target,
                    rank,
                    term,
                    "YES" if correct else "NO"
                ])

            
            for k in K_VALUES:
                hits[k] += recall_at_k(retrieved_terms, target, k)

        
        recall_writer.writerow([
            mode.upper(),
            f"{hits[1] / len(EVAL_DATA):.2f}",
            f"{hits[3] / len(EVAL_DATA):.2f}",
            f"{hits[5] / len(EVAL_DATA):.2f}",
        ])

        print("\nRecall Summary:")
        for k in K_VALUES:
            print(f"  Recall@{k}: {hits[k] / len(EVAL_DATA):.2f}")

    driver.close()
    recall_csv.close()
    qual_csv.close()

    print("\n==============================")
    print(" Tables generated successfully ")
    print("==============================")
    print("recall_summary.csv")
    print("qualitative_examples.csv")


if __name__ == "__main__":
    main()
