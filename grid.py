import copy
from collections import defaultdict
import os
import numpy as np


def _make_byte_array_readable(byte_array):
    return ''.join([b_char.decode('utf-8') for b_char in byte_array])


def _make_byte_array_list_readable(byte_array_list):
    return [_make_byte_array_readable(byte_array) for byte_array in byte_array_list]


def _make_grid_readable(grid):
    # ChatGPT wrote this
    # Create a list to hold the formatted strings for each row
    readable_rows = []
    # Iterate through each row of the original grid
    for i in range(grid.shape[0]):
        # Join the characters in the row into a single string with spaces
        row_string = '[' + ' '.join(grid[i, j].decode('utf-8') for j in range(grid.shape[1])) + ']'
        readable_rows.append(row_string)  # Append the row string to the list
    # Join all rows with a newline character for readability
    return '\n'.join(readable_rows)


class Grid:
    # CAN ONLY HANDLE SQUARE GRIDS FOR NOW
    def __init__(self, main_dict_fp, grid_size=5):
        self.main_dict_fp = main_dict_fp
        with open(main_dict_fp) as main_dict:
            self.dict = defaultdict(list)
            dict_entries = [dict_entry.strip("\n") for dict_entry in main_dict.readlines()]
            for dict_entry in dict_entries:
                self.dict[len(dict_entry)].append(np.array(list(dict_entry), dtype='S1'))
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype='S1')
        # Set the order of the entries.
        self.entry_order = [f"{i}-{letter}" for i in range(self.grid_size) for letter in ['a', 'd']]
        self.entry_filled = [False for _ in self.entry_order]
        self.entry_index = 0
        self.completed_grids = set()

    def __str__(self):
        return str(_make_grid_readable(self.grid))

    def __deepcopy__(self, memo):
        # Create a new instance
        grid_copy = Grid(main_dict_fp=copy.deepcopy(self.main_dict_fp, memo),
                         grid_size=copy.deepcopy(self.grid_size, memo))
        # Recursively deep copy all the attributes
        grid_copy.dict = copy.deepcopy(self.dict, memo)
        grid_copy.grid = copy.deepcopy(self.grid, memo)
        grid_copy.entry_order = copy.deepcopy(self.entry_order, memo)
        grid_copy.entry_filled = copy.deepcopy(self.entry_filled, memo)
        grid_copy.entry_index = copy.deepcopy(self.entry_index, memo)
        grid_copy.completed_grids = copy.deepcopy(self.completed_grids, memo)
        return grid_copy

    def __hash__(self):
        return hash(self.grid.tobytes())

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        return (self.grid_size == other.grid_size and
                np.array_equal(self.grid, other.grid) and
                self.entry_order == other.entry_order and
                self.entry_filled == other.entry_filled and
                self.entry_index == other.entry_index)

    # Gets the current index and direction
    def get_curr_i_dir(self):
        return self.entry_order[self.entry_index]

    # Gets the reference to the straight defined by the index and direction provided. It's not a deep copy
    def get_straight(self, i_dir):
        index, direction = i_dir.split("-")
        index = int(index)
        if direction == "a":
            return self.grid[index]
        elif direction == "d":
            return self.grid[:, index]
        else:
            raise Exception(f"Invalid i_dir: {i_dir}")

    def get_possible_words(self, straight):
        words_to_check = self.dict[len(straight)]

        # A helper function to see if two words match
        def match(straight, word_to_check):
            for straight_char, word_to_check_char in zip(straight, word_to_check):
                if straight_char and straight_char != word_to_check_char:
                    return False
            return True

        return [word_to_check for word_to_check in words_to_check if match(straight, word_to_check)]

    # Sets the straight mentioned to the possible_word
    def set_straight(self, possible_word, i_dir, i_dir_action):
        index, direction = i_dir.split("-")
        index = int(index)
        if direction == "a":
            self.grid[index] = possible_word
        elif direction == "d":
            self.grid[:, index] = possible_word
        else:
            raise Exception(f"Invalid i_dir: {i_dir}")
        # Update the i_dir according to the action
        if i_dir_action == "update":
            self.entry_filled[self.entry_index] = True
            # Find the next i_dir
            while self.entry_index < len(self.entry_order):
                self.entry_index += 1
                # If (a) we fill the grid or (b) we find the next valid entr_index
                if self.entry_index == len(self.entry_order) or not self.entry_filled[self.entry_index]:
                    break
            # This should only get here if (a) fill the grid or (b) we find the next valid entry_index
        elif i_dir_action == "revert":
            # Return it to the original state, i.e. this row being unfilled and the entry index lowered to before
            # We can only remove the entry_index if we're still in range
            if self.entry_index < len(self.entry_order):
                self.entry_filled[self.entry_index] = False
            self.entry_index = self.entry_order.index(i_dir)
        else:
            raise Exception(f"Invalid i_dir_action: {i_dir_action}")

    # Recursive function that fills the grid
    def fill_grid(self):
        # Successful exit condition
        if self.entry_index == len(self.entry_order):
            self.completed_grids.add(copy.deepcopy(self))
            return
        # Get the current index and direction for the unsuccessful exit condition
        curr_i_dir = self.get_curr_i_dir()
        curr_straight = self.get_straight(curr_i_dir).copy()  # Avoid weird reference semantics with self.grid
        possible_words = self.get_possible_words(curr_straight)
        # Unsuccessful exit condition
        if not len(possible_words):
            return
        # DFS
        for possible_word in possible_words:
            self.set_straight(possible_word, curr_i_dir, i_dir_action="update")
            self.fill_grid()
            self.set_straight(curr_straight, curr_i_dir, i_dir_action="revert")

    # Print out the completed grids
    def print_completed_grids(self):
        print(f"====================There are {len(self.completed_grids)} grids====================\n")
        for completed_grid in self.completed_grids:
            print(str(completed_grid) + "\n")


if __name__ == "__main__":
    test_grid = Grid(main_dict_fp=os.path.join("dictionaries", "small_5_dict.txt"), grid_size=5)
    test_grid.fill_grid()
    test_grid.print_completed_grids()
