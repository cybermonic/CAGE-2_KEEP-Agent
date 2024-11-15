### Model weights

The weights in this folder will work in the `keep_agent`. Models were trained against the BLine and Meander agents for 10,000 epochs, each consisting of 100 episodes. The `inductive_agent` used the default rewards, while the `high_precision` agent (will be available soon) used reward shaping to penalize incorrect Restore and Remove actions. Both agents get about -58 points across 30,50, and 100-step episodes.

The `random_topology` agent is optimized for MaTrEx v3. There, in 100-step games, it scores approximately -20 and -15 against the BLine and Meander agents, respectively. On normal CAGE-2, it scores approximately -200 in the same tests. The differences likely stem from the lack of servers in the MaTrEx environments, making `Remove` more valuable and utilized.