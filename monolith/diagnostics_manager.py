"""
Diagnostics Manager - Auto-generates adaptive diagnostics for LLM prompts
Tracks failures and generates context for next iteration
"""

import os
from typing import Dict, List, Any
from datetime import datetime


class DiagnosticsManager:
    """Manages diagnostic context for adaptive prompts"""
    
    def __init__(self, diagnostics_file: str = "diagnostics.txt"):
        self.diagnostics_file = diagnostics_file
        self.manual_notes_marker = "# MANUAL_NOTES_START"
        self.manual_notes = []
        self._load_manual_notes()
    
    def _load_manual_notes(self):
        """Load any existing manual notes from diagnostics file"""
        if not os.path.exists(self.diagnostics_file):
            return
        
        try:
            with open(self.diagnostics_file, 'r') as f:
                content = f.read()
            
            if self.manual_notes_marker in content:
                parts = content.split(self.manual_notes_marker)
                if len(parts) > 1:
                    self.manual_notes = parts[1].strip().split('\n')
        except Exception:
            pass
    
    def update_diagnostics(
        self,
        iteration: int,
        recent_failures: List[str],
        failure_frequencies: Dict[str, int],
        active_adaptations: Dict[str, str],
        additional_notes: List[str] = None
    ):
        """
        Update diagnostics file with current state
        
        Args:
            iteration: Current iteration number
            recent_failures: List of recent failure tags with descriptions
            failure_frequencies: Dict mapping failure tags to counts
            active_adaptations: Dict mapping failure tags to adaptation hints
            additional_notes: Optional list of additional notes
        """
        timestamp = datetime.now().isoformat()
        
        lines = [
            "# DIAGNOSTICS (auto-generated)",
            f"timestamp: {timestamp}",
            f"iteration: {iteration}",
            "",
            "recent_failures:"
        ]
        
        if recent_failures:
            for failure in recent_failures[-10:]:  # Last 10 failures
                lines.append(f"  - {failure}")
        else:
            lines.append("  - None")
        
        lines.append("")
        lines.append("failure_frequencies:")
        if failure_frequencies:
            for tag, count in sorted(failure_frequencies.items(), key=lambda x: -x[1]):
                lines.append(f"  {tag}: {count}")
        else:
            lines.append("  None")
        
        lines.append("")
        lines.append("active_adaptations:")
        if active_adaptations:
            for tag, hint in active_adaptations.items():
                lines.append(f"  {tag}: \"{hint}\"")
        else:
            lines.append("  None")
        
        lines.append("")
        lines.append("notes:")
        
        # Add standard notes
        standard_notes = [
            "- Enforce get_consensus_signal singular method.",
            "- Synthesize high/low if missing.",
            "- Strip example usage blocks.",
            "- Ensure AdaptiveTradingStrategy class present.",
            "- Return list-of-dicts from generate_signals.",
        ]
        
        for note in standard_notes:
            lines.append(f"  {note}")
        
        # Add additional notes
        if additional_notes:
            for note in additional_notes:
                lines.append(f"  {note}")
        
        # Preserve manual notes
        lines.append("")
        lines.append(self.manual_notes_marker)
        if self.manual_notes:
            for note in self.manual_notes:
                if note.strip():
                    lines.append(note)
        else:
            lines.append("  (Your custom notes remain here)")
        
        # Write to file
        try:
            with open(self.diagnostics_file, 'w') as f:
                f.write('\n'.join(lines))
        except Exception as e:
            print(f"Warning: Failed to write diagnostics: {e}")
    
    def get_active_adaptations(self, failure_frequencies: Dict[str, int]) -> Dict[str, str]:
        """
        Generate adaptation hints based on failure frequencies
        
        Returns:
            Dict mapping failure tags to adaptation hints
        """
        adaptations = {}
        
        if failure_frequencies.get('SYNTAX_ERROR', 0) > 0:
            adaptations['SYNTAX_ERROR'] = \
                "Simplify structure; remove markdown fences; ensure class present."
        
        if failure_frequencies.get('LOW_SIGNAL', 0) > 2:
            adaptations['LOW_SIGNAL'] = \
                "Add second signal path (volatility + momentum). Lower thresholds."
        
        if failure_frequencies.get('INTERFACE_ERROR', 0) > 0:
            adaptations['INTERFACE_ERROR'] = \
                "Ensure list-of-dicts convertible: include symbol, action, price, strategy."
        
        if failure_frequencies.get('LOW_ALPHA', 0) > 0:
            adaptations['LOW_ALPHA'] = \
                "Increase confidence calibration; diversify symbols."
        
        if failure_frequencies.get('RUNTIME_FAIL', 0) > 1:
            adaptations['RUNTIME_FAIL'] = \
                "Remove fragile dynamic imports; simplify signal generation logic."
        
        return adaptations
    
    def read_diagnostics(self) -> str:
        """Read current diagnostics content"""
        if not os.path.exists(self.diagnostics_file):
            return self._get_fallback_diagnostics()
        
        try:
            with open(self.diagnostics_file, 'r') as f:
                return f.read()
        except Exception:
            return self._get_fallback_diagnostics()
    
    def _get_fallback_diagnostics(self) -> str:
        """Return fallback diagnostics if file doesn't exist"""
        return """# DIAGNOSTICS (fallback)
timestamp: INIT
iteration: 0

recent_failures:
  - None

failure_frequencies:
  None

active_adaptations:
  None

notes:
  - Contracts enforced.
  - Await first iteration to populate diagnostics.

# MANUAL_NOTES_START
  - You can add manual overrides here; preserved across updates.
"""


# For standalone testing
if __name__ == "__main__":
    dm = DiagnosticsManager("/tmp/test_diagnostics.txt")
    
    # Update with sample data
    dm.update_diagnostics(
        iteration=5,
        recent_failures=[
            "SYNTAX_ERROR: strategies.py unexpected EOF",
            "LOW_SIGNAL: trades=3 diversity=0.50 threshold=6"
        ],
        failure_frequencies={
            'SYNTAX_ERROR': 4,
            'LOW_SIGNAL': 7,
            'INTERFACE_ERROR': 2,
            'LOW_ALPHA': 1
        },
        active_adaptations=dm.get_active_adaptations({
            'SYNTAX_ERROR': 4,
            'LOW_SIGNAL': 7,
            'INTERFACE_ERROR': 2,
            'LOW_ALPHA': 1
        }),
        additional_notes=["Testing diagnostics generation"]
    )
    
    print("Generated diagnostics:")
    print(dm.read_diagnostics())
