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


if __name__ == "__main__":
    unittest.main()
