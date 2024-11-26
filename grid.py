import copy
from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm


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


# A helper function to see if two words match
def match(straight, word_to_check):
    for straight_char, word_to_check_char in zip(straight, word_to_check):
        if straight_char and straight_char != word_to_check_char:
            return False
    return True


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
        self.completed_grids = []

        # Set verbose commands
        self.verbose = defaultdict(bool)

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
        if self.verbose['Print completed words']:
            print(f"Checking {len(words_to_check)} words")
        valid_words = [word_to_check for word_to_check in words_to_check if match(straight, word_to_check)]
        if self.verbose['Print completed words']:
            print(f"Found {len(valid_words)} words compatible with {straight} \n")
        # TODO continue from here, this method needs to be made faster
        return valid_words

    # Sets the straight but doesn't do any updates
    def _set_straight(self, possible_word, i_dir, check_possible=False):
        if check_possible and not match(self.get_straight(i_dir=i_dir), word_to_check=possible_word):
            raise Exception(f"Straight {possible_word} given i_dir {i_dir} does not fit in the following grid: \n"
                            f"{_make_grid_readable(self.grid)}")
        index, direction = i_dir.split("-")
        index = int(index)
        if direction == "a":
            self.grid[index] = possible_word
        elif direction == "d":
            self.grid[:, index] = possible_word
        else:
            raise Exception(f"Invalid i_dir: {i_dir}")

    # Sets the straight mentioned to the possible_word
    def set_straight(self, possible_word, i_dir, i_dir_action):
        self._set_straight(possible_word=possible_word, i_dir=i_dir, check_possible=False)
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
    def fill_grid(self, max_completed_grids=100):
        # Successful exit condition
        if len(self.completed_grids) == max_completed_grids:
            return
        if self.entry_index == len(self.entry_order):
            self.completed_grids.append(self.grid.copy() )
            if self.verbose['Print when found grid']:
                print(f"Found grid. \n{_make_grid_readable(self.grid.copy() )}")
            return
        # Get the current index and direction for the unsuccessful exit condition
        curr_i_dir = self.get_curr_i_dir()
        curr_straight = self.get_straight(curr_i_dir).copy()  # Avoid weird reference semantics with self.grid
        possible_words = self.get_possible_words(curr_straight)
        # Unsuccessful exit condition
        if not len(possible_words):
            return
        # DFS
        words_iterator = tqdm(possible_words) if self.entry_index == 0 else possible_words
        for possible_word in words_iterator:
            self.set_straight(possible_word, curr_i_dir, i_dir_action="update")
            self.fill_grid()
            self.set_straight(curr_straight, curr_i_dir, i_dir_action="revert")

    # Manually fill out straights in the grid
    # straights must be a list of tuples, consisting of direction and word
    def manually_set_straights(self, straights):
        for word, i_dir in straights:
            word_array = np.array(list(word), dtype="S1")
            self._set_straight(possible_word=word_array, i_dir=i_dir, check_possible=True)
            # Now that we manually set the straights, we don't want them to be able to be removed
            i_dir_index = self.entry_order.index(i_dir)
            if i_dir_index == -1:
                raise Exception(f"Illegal i_dir {i_dir} for {word}")
            del self.entry_order[i_dir_index]
            del self.entry_filled[i_dir_index]

    # Helper method to get the output for printing grids and saving grids
    def _get_completed_grids_output(self):
        output = [f"====================There are {len(self.completed_grids)} grids====================\n"]
        for completed_grid in self.completed_grids:
            output.append(_make_grid_readable(completed_grid) + "\n\n")
        return output

    # Print out the completed grids
    def print_completed_grids(self):
        print("".join(self._get_completed_grids_output() ) )

    # Saves the completed grids
    # file_name can be a string or a tuple consisting of the straights and the dict name
    # They will always save in the finished_crosswords directory
    def save_completed_grids(self, file_name, print_grids=False):
        output = self._get_completed_grids_output()
        if print_grids:
            print("".join(output) )
        if isinstance(file_name, tuple):
            file_straights, file_dict_name = file_name
            file_straights = "__".join(f"{file_word}_{file_i_dir}" for file_word, file_i_dir in file_straights)
            file_name = f"{file_straights}__{file_dict_name}"
        elif not isinstance(file_name, str):
            raise Exception("file_name must be a string or a tuple of straights and dict_name")
        with open(os.path.join("finished_crosswords", file_name), "w") as output_file:
            output_file.writelines(output)


if __name__ == "__main__":
    straights = [("wordy", "1-d")]
    dict_name = "wordle_small_5_dict.txt"
    test_grid = Grid(main_dict_fp=os.path.join("dictionaries", dict_name), grid_size=5)
    test_grid.verbose['Print when found grid'] = True
    test_grid.manually_set_straights(straights=straights)
    test_grid.fill_grid(max_completed_grids=1)
    test_grid.save_completed_grids(file_name=(straights, dict_name), print_grids=True)
