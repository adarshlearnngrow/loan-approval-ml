"""
Input Validators
"""


def validate_ratios(approved: float, refused: float, cancelled: float, 
                   unused: float, total_apps: int) -> tuple:
    """
    Validate that application ratios sum to approximately 1.0
    
    Returns:
        tuple: (is_valid, message)
    """
    if total_apps == 0:
        return True, ""
    
    ratio_sum = approved + refused + cancelled + unused
    
    if abs(ratio_sum - 1.0) > 0.06:
        return False, f"Ratios sum to {ratio_sum:.2f} — they should ideally sum to 1.0"
    
    return True, ""
