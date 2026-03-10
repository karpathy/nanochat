"""
Tests for ContextWindowManager
"""
import pytest
from nanochat.context_window import ContextWindowManager


class TestContextWindowManager:
    """Test context window truncation logic."""
    
    def test_no_truncation_needed(self):
        """When conversation fits, return unchanged."""
        manager = ContextWindowManager(max_length=100, reserved_tokens=20)
        tokens = list(range(50))
        
        result = manager.truncate(tokens, [])
        
        assert result == tokens
    
    def test_simple_truncation(self):
        """Truncate from beginning when no turn boundaries."""
        manager = ContextWindowManager(max_length=100, reserved_tokens=20)
        tokens = list(range(150))
        
        result = manager.truncate(tokens, [])
        
        # Should have 80 tokens (100 - 20 reserved)
        assert len(result) <= 80
        # Should keep most recent tokens
        assert result[-1] == tokens[-1]
    
    def test_preserve_system_tokens(self):
        """System tokens should always be preserved."""
        system = [0, 1, 2]  # BOS + system
        manager = ContextWindowManager(
            max_length=100,
            reserved_tokens=20,
            system_tokens=system,
        )
        tokens = system + list(range(3, 150))
        
        result = manager.truncate(tokens, [])
        
        # System tokens preserved
        assert result[:3] == system
    
    def test_turn_boundary_detection(self):
        """Correctly identify turn boundaries."""
        manager = ContextWindowManager(max_length=1000)
        
        # Simulated conversation: bos + user1 + assistant1 + user2 + assistant2
        USER_START, USER_END = 100, 101
        ASST_START, ASST_END = 200, 201
        
        tokens = [
            0,  # bos
            USER_START, 10, 20, 30, USER_END,  # user turn 1
            ASST_START, 40, 50, ASST_END,  # assistant turn 1
            USER_START, 60, 70, USER_END,  # user turn 2
            ASST_START, 80, 90, ASST_END,  # assistant turn 2
        ]
        
        boundaries = manager.get_turn_boundaries(
            tokens,
            user_start=USER_START,
            user_end=USER_END,
            assistant_start=ASST_START,
            assistant_end=ASST_END,
        )
        
        assert len(boundaries) == 2
        # First turn: user1 + assistant1
        assert boundaries[0] == (1, 10)
        # Second turn: user2 + assistant2  
        assert boundaries[1] == (10, 18)
    
    def test_truncate_keeps_recent_turns(self):
        """When truncating, keep most recent turns."""
        manager = ContextWindowManager(
            max_length=20,
            reserved_tokens=5,
            system_tokens=[0],
        )
        
        # Turn boundaries: [start, end)
        turns = [
            (1, 8),   # Turn 1: 7 tokens
            (8, 15),  # Turn 2: 7 tokens  
            (15, 22), # Turn 3: 7 tokens
        ]
        
        tokens = list(range(22))
        
        result = manager.truncate(tokens, turns)
        
        # Should keep system + most recent turn(s)
        assert result[0] == 0  # System token
        # Most recent turn should be preserved
        assert 15 in result or 16 in result or 17 in result
    
    def test_incomplete_turn_ignored(self):
        """Incomplete turns (no assistant_end) should not be counted."""
        manager = ContextWindowManager(max_length=1000)
        
        USER_START, USER_END = 100, 101
        ASST_START, ASST_END = 200, 201
        
        tokens = [
            0,
            USER_START, 10, 20, USER_END,
            ASST_START, 30, ASST_END,  # Complete turn
            USER_START, 40, 50, USER_END,  # Incomplete - no assistant response
        ]
        
        boundaries = manager.get_turn_boundaries(
            tokens,
            user_start=USER_START,
            user_end=USER_END,
            assistant_start=ASST_START,
            assistant_end=ASST_END,
        )
        
        assert len(boundaries) == 1  # Only complete turn


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
