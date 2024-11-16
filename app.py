import asyncio
import plotly.express as px
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import numpy as np
import re
from skspatial.objects import Plane

# Define UI components
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file_input", "Upload FHI-aims output files", multiple=True),
        ui.input_text("atom_selection", "Select atoms (e.g., 1, 3-5, 8)", placeholder="Enter atoms"),
        ui.input_action_button("calculate_button", "Calculate Momentum")
    ),
    ui.card(
        output_widget("geometry_plot"),
        ui.output_text_verbatim("momentum_output"),
#        ui.output_text_verbatim("geometry_debug"),
#        ui.output_text_verbatim("forces_debug")
    )
)

# Helper function to parse atom selection input
def parse_atom_selection(selection_str):
    atoms = []
    ranges = selection_str.split(',')
    for r in ranges:
        if '-' in r:
            try:
                start, end = map(int, r.split('-'))
                atoms.extend(range(start, end + 1))
            except ValueError:
                return []  # Return empty list if parsing fails
        else:
            try:
                atoms.append(int(r))
            except ValueError:
                return []  # Return empty list if parsing fails
    return atoms

# Helper function to locate and extract the first geometry data from a file
def extract_geometry(file_content):
    geometry_block = []
    parsing_geometry = False

    lines = file_content.splitlines()
    for idx, line in enumerate(lines):
        if "Parsing geometry.in (first pass over file, find array dimensions only)" in line and not geometry_block:
            parsing_geometry = True
            # Skip empty lines before geometry data starts
            while idx + 1 < len(lines) and lines[idx + 1].strip() == "":
                idx += 1
            continue

        if parsing_geometry:
            if line.lstrip().startswith("atom"):
                geometry_block.append(line)
            elif line.strip() == "" and geometry_block:
                parsing_geometry = False
                break  # Stop after finding the first complete geometry block

    return geometry_block

# Helper function to locate and extract the first force data from a file
def extract_forces(file_content):
    forces_block = []
    parsing_forces = False

    lines = file_content.splitlines()
    for idx, line in enumerate(lines):
        if "Total atomic forces (unitary forces cleaned) [eV/Ang]:" in line and not forces_block:
            parsing_forces = True
            # Skip empty lines before forces data starts
            while idx + 1 < len(lines) and lines[idx + 1].strip() == "":
                idx += 1
            continue

        if parsing_forces:
            if line.lstrip().startswith("|"):
                forces_block.append(line)
            # Stop if we encounter an empty line or if the number of lines matches the number of atoms
            if line.strip() == "" or len(forces_block) == len(extract_geometry(file_content)):
                parsing_forces = False
                break

    return forces_block

# Function to calculate the total momentum
def calculate_momentum(geometry, forces, selected_atoms):
    # Calculate the position of Cr and the center of selected six Carbon atoms
    cr_position = None
    carbon_positions = []

    for i, line in enumerate(geometry):
        parts = line.split()
        x, y, z = map(float, parts[1:4])
        element = parts[4]
        if element == "Cr":
            cr_position = np.array([x, y, z])
        elif element == "C" and i + 1 in selected_atoms:
            carbon_positions.append(np.array([x, y, z]))

    if len(carbon_positions) < 6:
        raise ValueError("Not enough Carbon atoms for axis calculation.")

    if cr_position is None:
        raise ValueError("Cr needed to specify the axis")

    # Calculate the center of the six Carbon atoms
    carbon_center = np.mean(carbon_positions, axis=0)

    axis_direction = carbon_center - cr_position
    axis_direction /= np.linalg.norm(axis_direction)  # Normalize the axis direction

    # Calculate the total momentum with respect to the axis
    total_momentum = 0.0
    for i, force_line in enumerate(forces):
        if i + 1 in selected_atoms:
            _, fx, fy, fz = map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", force_line))
            force_vector = np.array([fx, fy, fz])
            position_vector = np.array([float(x) for x in geometry[i].split()[1:4]]) - cr_position
            # Calculate the component of the momentum perpendicular to the axis
            perpendicular_component = np.cross(position_vector, force_vector)
            momentum_along_axis = np.dot(perpendicular_component, axis_direction)
            total_momentum += momentum_along_axis
    return total_momentum

# Server logic
def server(input, output, session):
    @render_widget
    @reactive.event(input.calculate_button)
    def geometry_plot():
        if input.file_input() is None:
            return px.scatter_3d(width = 800, height = 600)

        # Load the first file for simplicity (extendable to multiple files)
        file_info = input.file_input()[0]
        with open(file_info["datapath"], 'r') as f: file_content = f.read()
        geometry = extract_geometry(file_content)

        # Parse atom positions
        if not geometry:
            return px.scatter_3d(width = 800, height = 600)

        atoms = []
        for line in geometry:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            element = parts[4]
            atoms.append((x, y, z, element))

        # Create 3D scatter plot of atoms
        df = {'x': [atom[0] for atom in atoms], 'y': [atom[1] for atom in atoms], 'z': [atom[2] for atom in atoms], 'element': [atom[3] for atom in atoms]}
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='element', title='Geometry Plot', width = 800, height = 600)
        return fig

    @output
    @render.text
    @reactive.event(input.calculate_button)
    async def momentum_output():
        if input.file_input() is None or input.atom_selection() == "":
            return "Please upload files and select atoms."

        # Load the first file for simplicity (extendable to multiple files)
        file_info = input.file_input()[0]
        with open(file_info["datapath"], 'r') as f: file_content = f.read()
        geometry = extract_geometry(file_content)
        forces = extract_forces(file_content)
        selected_atoms = parse_atom_selection(input.atom_selection())

        if not selected_atoms:
            return "Invalid atom selection. Please enter valid atom indices."

        # Calculate total momentum
        if not forces:
            return "No forces data found."
        try:
            total_momentum = calculate_momentum(geometry, forces, selected_atoms)
            return f"Total Momentum: {total_momentum} (eV)"
        except ValueError as e:
            return str(e)
# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
