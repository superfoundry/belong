"""Tests never call the network (CLAUDE.md): whatever the shell exports, the demo's Matchmaker runs on the rules here
unless a test passes a stub judge (DemoSession(matchmaker=judge)) or sets the choice itself, so a key in the
environment never makes a test ask a model."""
import os

os.environ["BELONG_MATCHMAKER"] = "rules"
