from neo4j import GraphDatabase
from retrieve_snomed import retrieve_snomed_matches
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import csv


# PATIENT_TURNS = ["Hey, Dr. Sinclair. Thanks for chatting with me. Lately, I've been feeling this heaviness in my chest. It's like there's a weight on me that won't go away. It's been really hard to shake off, you know?", 
#                  "It kind of hit me out of the blue a few weeks ago. There was this event that happened that just left me feeling off. Ever since then, it's like I can't shake this feeling of being down all the time. It's been tough to deal with, you know?", 
#                  "It's hard to put into words, but that event really hit me hard emotionally. It's like it knocked the wind out of me, you know? It made me question a lot of things about myself and my life. I guess it just left me feeling lost and unsure about everything.", 
#                  "Yeah, it's been tough. I've been having trouble sleeping, you know? My mind just races at night, and I can't seem to switch it off. And my appetite has been all over the place. Some days I don't feel like eating at all, and other days I can't stop myself from snacking. It's like my body's out of whack, and I just feel drained all the time.", 
#                  "Yeah, it's been rough. I used to love going out with friends and doing things I enjoyed, but lately, I just haven't had the energy or motivation for it. It's like everything feels like a chore, you know? And I feel like I've been withdrawing from people because I don't want to bring them down with my own issues. It's like I'm just going through the motions without really feeling anything.",
#                  "It's been tough, you know? I used to feel pretty confident in myself and my abilities, but lately, it's like that confidence has just disappeared. I find myself doubting everything I do and feeling like I'm not good enough. It's like I've lost sight of who I am and what I want out of life. Everything just feels kind of bleak and uncertain right now.", 
#                  "I guess I've been telling myself a lot of negative things lately. Like, I keep thinking that I'm a failure, that I'm not good enough, that I'm just going to mess everything up. It's like these thoughts just spiral in my head, you know? And they make me feel so down and hopeless. It's like I can't escape this cycle of self-criticism and negativity.", 
#                  "I think a lot of it stems from that event that happened. It made me question my worth and abilities in a way I hadn't before. It's like it unearthed all these insecurities and doubts that I didn't even know were there. And now, it's like those thoughts have taken over, you know? It's like I can't escape them no matter how hard I try.",
#                  "I appreciate your support, Dr. Sinclair. The event that happened made me feel like I had failed in some way, like I wasn't good enough to handle it. It brought up all these feelings of inadequacy and self-doubt that have been weighing me down ever since. It's like it shattered this image I had of myself and left me feeling lost and unsure. ", 
#                  "Yeah, I think I'm ready to start working on reframing those negative thoughts. It's been exhausting carrying all this self-doubt around, and I know I need to find a way to let go of it. I'm willing to try to challenge those beliefs and see things from a different perspective. I just want to feel like myself again, you know?" 
                #  ]
PATIENT_TURNS = ["I feel like my coursework is piling up faster than I can keep track of it, and every time I sit down to work, my mind jumps to everything else that could go wrong.",
                "I keep telling myself I just need to push through this week, but then another deadline appears, and it feels like there’s no real break ahead.", 
                "I’m overwhelmed not just by school, but by life in general—finances, future plans, and the pressure to “figure things out” all seem to hit at once.", 
                "When I fall behind even a little, I start spiraling and thinking it means I’m failing at more than just this one class.",
                "I notice that I avoid starting assignments because I’m afraid I won’t do them well enough, which only makes the stress worse.",
                "I feel guilty when I rest, but exhausted when I work, and I don’t know how to find a balance that doesn’t feel like I’m neglecting something important.",
                "Sometimes I wonder if everyone else is coping better than I am, or if they’re just better at hiding how overwhelmed they feel.",
                "I’m worried that if I keep going like this, I’ll burn out completely, but I don’t know what slowing down would even look like right now."
                "I want to feel more in control of my time and my emotions, but everything feels reactive, like I’m constantly putting out fires.",
                "I guess I’m here because I don’t want to keep carrying all of this alone, and I need help figuring out how to move forward without feeling so crushed all the time."]

def fetch_term(driver, code):
    """Fetch SNOMED term given a code."""
    q = "MATCH (c:Concept {code: $code}) RETURN c.term AS term"
    with driver.session() as session:
        r = session.run(q, code=code).single()
        return r["term"] if r else None


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("\n============= TURN-BY-TURN SNOMED EXTRACTION =============\n")

    # File to save results
    csv_file = open("bioreddit_turn_results.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["Turn", "User Text", "SNOMED Term", "Code", "Score", "Relation Type", "Relation Target Code", "Relation Target Term"])

    for i, turn in enumerate(PATIENT_TURNS, start=1):

        print(f"\n\n==================== TURN {i} ====================\n")
        print(f"User said:\n{turn}\n")

        results = retrieve_snomed_matches(driver, turn, k=5)

        print("Top 5 SNOMED Matches:")
        print("---------------------------------------------")

        for r in results:

            # If no match
            if r["term"] == "No matches found":
                print("No matches found\n")
                writer.writerow([i, turn, None, None, None, None, None, None])
                continue


            code = r["code"]
            term = r["term"]
            score = round(r["score"], 4)

            print(f"• {term}  (code: {code}, score: {score})")

       
            if r["relations"]:
                for rel in r["relations"]:
                    rel_type = rel["type"]
                    target_code = rel["target"]

                    target_term = fetch_term(driver, target_code)

                    print(f"     → {rel_type} → {target_term} (code: {target_code})")

                    writer.writerow([i, turn, term, code, score, rel_type, target_code, target_term])
            else:
                writer.writerow([i, turn, term, code, score, None, None, None])

        print("\n---------------------------------------------")

    csv_file.close()
    driver.close()

    print("\n\nResults saved to snomed_turn_results.csv\n")


if __name__ == "__main__":
    main()
