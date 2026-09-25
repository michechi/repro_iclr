from typing import List, Union

def check_lag(seq: List, keys: List, lags: Union[int, List[int]] = 6, 
              overflow_strategy: str = 'repeat_last') -> Union[List[List], bool]:
    """
    Find chains of elements at specific distances (fixed or variable).
    
    Args:
        seq: sequence of elements
        keys: elements to look for in the sequence
        lags: can be:
            - int: fixed distance between all elements
            - List[int]: variable distances [d1, d2, d3, ...]
                where d1 = distance between elem 1 and 2
                      d2 = distance between elem 2 and 3, etc.
        overflow_strategy: what to do when there are more elements than lags:
            - 'repeat_last': reuse the last lag for subsequent elements
            - 'cycle': start again from the first lag (cyclic pattern)
            - 'any': any distance is fine after the specified lags
            - 'strict': accept only chains as long as the number of lags + 1
    
    Returns:
        False if no valid pattern is found
        List of lists with the patterns found
    
    Examples:
        >>> seq = ['A', 'X', 'X', 'X', 'X', 'X', 'X', 'B', 'Y', 'Y', 'Y', 'C', 'Z', 'Z', 'D']
        >>> keys = ['A', 'B', 'C', 'D']
        >>> 
        >>> # Fixed lag (original behaviour)
        >>> check_lag(seq, keys, lags=3)
        False  # No pattern with fixed lag 3
        >>> 
        >>> # Variable lags [7, 4, 3]
        >>> check_lag(seq, keys, lags=[7, 4, 3])
        [['A', 'B', 'C', 'D']]  # A->B dist 7, B->C dist 4, C->D dist 3
    """
    
    # Convert a single lag into a list for uniformity
    if isinstance(lags, int):
        lag_list = [lags]
        is_fixed_lag = True
    else:
        lag_list = lags
        is_fixed_lag = False

    # Input validation
    if not lag_list:
        raise ValueError("The lag list cannot be empty")
    
    if lags == 0:
        return [elem for elem in seq if elem in keys]
    
    # Find all positions of the key elements
    key_positions = [(i, elem) for i, elem in enumerate(seq) if elem in keys]
    
    if not key_positions:
        return False
    
    chains = []
    used_positions = set()
    
    # Helper function to get the lag for a given position in the chain
    def get_lag_for_position(chain_index: int) -> Union[int, None]:
        """
        Return the lag to use for position chain_index in the chain.
        chain_index 0 means we are looking for the second element (lag between 1st and 2nd)
        """
        if is_fixed_lag:
            return lag_list[0]
        
        if chain_index < len(lag_list):
            return lag_list[chain_index]
        
        # Overflow handling: we have more elements than specified lags
        if overflow_strategy == 'repeat_last':
            return lag_list[-1]
        elif overflow_strategy == 'cycle':
            return lag_list[chain_index % len(lag_list)]
        elif overflow_strategy == 'any':
            return None  # Signal that any distance is fine
        elif overflow_strategy == 'strict':
            return False  # Do not continue the chain
        else:
            raise ValueError(f"Invalid overflow strategy: {overflow_strategy}")
    
    # For each possible starting position
    for start_pos, start_elem in key_positions:
        if start_pos in used_positions:
            continue
        
        # Build the chain
        chain = [start_elem]
        chain_positions = [start_pos]
        current_pos = start_pos
        chain_index = 0  # Index to know which lag to use
        
        # Keep looking for elements according to the specified lags
        while True:
            # Get the lag for this position
            expected_lag = get_lag_for_position(chain_index)
            
            if expected_lag is False:  # 'strict' strategy - stop
                break
            
            found = False
            
            if expected_lag is None:  # 'any' strategy - accept any key element
                # Find the next key element at any distance
                for pos, elem in key_positions:
                    if pos > current_pos and elem in keys and pos not in chain_positions:
                        chain.append(elem)
                        chain_positions.append(pos)
                        current_pos = pos
                        found = True
                        break
            else:  # Specific lag
                next_pos = current_pos + expected_lag
                # Look for a key element at the expected position
                for pos, elem in key_positions:
                    if pos == next_pos and elem in keys:
                        chain.append(elem)
                        chain_positions.append(pos)
                        current_pos = next_pos
                        found = True
                        break
            
            if not found:
                break
            
            chain_index += 1
        
        # Save the chain if it has at least 2 elements (valid pattern)
        if len(chain) >= 2:
            # Check that it is not a subset of an existing chain
            is_subset = False
            for i, existing_chain in enumerate(chains):
                existing_positions = [p for p, _ in existing_chain]
                if set(chain_positions).issubset(set(existing_positions)):
                    is_subset = True
                    break
                # If this chain contains an existing one, replace it
                elif set(existing_positions).issubset(set(chain_positions)):
                    chains[i] = list(zip(chain_positions, chain))
                    used_positions.update(chain_positions)
                    is_subset = True
                    break
            
            if not is_subset:
                chains.append(list(zip(chain_positions, chain)))
                used_positions.update(chain_positions)
    
    if not chains:
        return False
    
    # Return only the lists of elements (without positions)

    return [[elem for _, elem in chain] for chain in chains]