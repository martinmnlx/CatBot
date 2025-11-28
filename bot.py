import argparse
import time
from cat_env import make_env
from training import train_bot
from utility import play_q_table

def main():
    parser = argparse.ArgumentParser(description='Train and play Cat Chase bot')
    parser.add_argument('--cat', 
                       choices=['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi', 'trainer'],
                       default='batmeow',
                       help='Type of cat to train against (default: mittens)')
    parser.add_argument('--render', 
                       type=int,
                       default=-1,
                       help='Render the environment every n episodes (default: -1, no rendering)')
    
    args = parser.parse_args()
    
    # Train the agent
    print(f"\nTraining agent against {args.cat} cat...")
    
    start_time = None
    if args.render == -1:
        start_time = time.time()

    q_table = train_bot(
        cat_name=args.cat,
        render=args.render
    )
    
    if start_time is not None:
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\nTraining complete in {training_duration:.2f} seconds! Starting game with trained bot...")
    else:
        print(f"\nTraining complete! Starting game with trained bot...")

    # --- Evaluation Runs ---
    num_runs = 20
    successes = 0
    total_moves = 0
    print(f"\n--- Running {num_runs} evaluation runs ---")
    print("Press Q to quit during a run.")

    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}:")
        env = make_env(cat_type=args.cat)
        # Use a slightly longer move_delay to make visualization clearer
        success, moves = play_q_table(env, q_table, max_steps=60, move_delay=0.05, window_title=f'Cat Chase - Evaluation Run {i+1}')
        if success:
            successes += 1
            total_moves += moves

    print("\n--- Evaluation Summary ---")
    success_rate = (successes / num_runs) * 100
    print(f"Success rate: {success_rate:.1f}% ({successes}/{num_runs})")
    if successes > 0:
        average_moves = total_moves / successes
        print(f"Average moves on success: {average_moves:.2f}")

if __name__ == "__main__":
    main()
