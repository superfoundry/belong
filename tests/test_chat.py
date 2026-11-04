"""The chat mockup (/chat) runs on its own session: one person, never touching the phones."""
from __future__ import annotations

import unittest

from belong import server


class ChatMockup(unittest.TestCase):
    def test_the_chat_session_is_separate_from_the_phones(self):
        chat = server.chat_session()
        self.assertIsNot(chat, server.SESSION)
        self.assertIs(server.chat_session(), chat)  # made once
        state = server.chat_state(chat)
        self.assertEqual(set(state), {"me", "agent_status"})
        self.assertNotIn("room", state)  # the person sees only their own state

    def test_only_a_persons_own_actions_are_allowed(self):
        self.assertNotIn("attack", server.CHAT_ACTIONS)  # presenter tools stay on the demo
        self.assertTrue(server.CHAT_ACTIONS <= set(server.ACTIONS))


class OpenDetailsInTheConnector(unittest.TestCase):
    def test_the_connector_reads_an_open_details_key_as_the_phone_does(self):
        """One rule for about:name on every page (the phone's is checked against schema.split_key in test_demo)."""
        import re
        from pathlib import Path
        web = Path(__file__).resolve().parent.parent / "belong" / "web"
        rule = lambda page: re.search(r"^const detailParts = .*$", (web / page).read_text(), flags=re.M).group(0)
        self.assertEqual(rule("chat.html"), rule("demo.html"))
        chat = (web / "chat.html").read_text()
        self.assertNotIn('"choice"', chat)  # no question has a fixed list of choices any more
        self.assertIn("What they'd like", chat)


if __name__ == "__main__":
    unittest.main()
