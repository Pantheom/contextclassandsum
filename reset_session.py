"""
reset_session.py
----------------
Quick tool to reset a session's summary and message pointer in Supabase,
or optionally wipe chat history to resolve context window overflow errors.

Usage:
    python reset_session.py
    python reset_session.py <uid>
    python reset_session.py <uid> --delete-history
    python reset_session.py --all
"""

import argparse
import os
import sys
import dotenv
from supabase import create_client


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dotenv.load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    client = create_client(url, key)

    parser = argparse.ArgumentParser(description="Reset Supabase session summary / chat history.")
    parser.add_argument("uid", nargs="?", help="The session/user UUID to reset.")
    parser.add_argument("--all", action="store_true", help="Reset all sessions.")
    parser.add_argument(
        "--delete-history",
        action="store_true",
        help="Also delete rows in chat_history for this session.",
    )
    args = parser.parse_args()

    # Determine which UIDs to reset
    if args.all:
        resp = client.table("context_classifier").select("uid").execute()
        uids = [r["uid"] for r in resp.data]
    elif args.uid:
        uids = [args.uid]
    else:
        # If no arguments provided, show existing sessions
        resp = client.table("context_classifier").select("uid, last_summarized_message_id").execute()
        if not resp.data:
            print("No sessions found in context_classifier.")
            return

        print("Active sessions found:")
        for idx, row in enumerate(resp.data, 1):
            uid = row["uid"]
            h_count = client.table("chat_history").select("id", count="exact").eq("uid", uid).execute()
            count = h_count.count if h_count.count is not None else len(h_count.data)
            print(f"  [{idx}] uid: {uid} | msgs: {count} | last_id: {row['last_summarized_message_id']}")

        print("\nSpecify a UID to reset, e.g.:")
        print(f"  python reset_session.py {resp.data[0]['uid']}")
        return

    for uid in uids:
        print(f"\nProcessing session: {uid}")

        if args.delete_history:
            client.table("chat_history").delete().eq("uid", uid).execute()
            print("  Deleted chat_history messages.")
            latest_id = 0
        else:
            # Find latest message id so summarizer skips past all existing messages
            h_resp = (
                client.table("chat_history")
                .select("id")
                .eq("uid", uid)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            latest_id = h_resp.data[0]["id"] if h_resp.data else 0

        # Update context_classifier
        client.table("context_classifier").upsert(
            {
                "uid": uid,
                "chat_summary": "",
                "last_summarized_message_id": latest_id,
            },
            on_conflict="uid",
        ).execute()

        print(f"  Summary cleared and last_summarized_message_id advanced to {latest_id}.")
        print("  -> Summarizer will start fresh from subsequent messages without token overflow.")


if __name__ == "__main__":
    main()
