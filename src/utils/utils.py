def transform_multi_choice_column_other(input_string):
    """
    Transform a string like "a; b; c; Other: x, y, z" into "a; b; c; x; y; z"
    
    Args:
        input_string (str): Input string with semicolon-separated items and optional "Other:" section
        
    Returns:
        str: Transformed string with all items semicolon-separated
    """
    if not input_string or not isinstance(input_string, str):
        return ""
    
    # Split on "Other:" to separate main items from other items
    parts = input_string.split("Other:")
    
    # Get the main part (before "Other:")
    main_part = parts[0].strip()
    
    # Start with main items (split by semicolon and clean up)
    if main_part:
        main_items = [item.strip() for item in main_part.split(";") if item.strip()]
    else:
        main_items = []
    
    # If there's an "Other:" section, process it
    if len(parts) > 1:
        other_part = parts[1].strip()
        if other_part:
            # Split by comma and clean up
            other_items = [item.strip() for item in other_part.split(",") if item.strip()]
            # Add other items to main items
            main_items.extend(other_items)
    
    # Join all items with semicolons
    return "; ".join(main_items)
