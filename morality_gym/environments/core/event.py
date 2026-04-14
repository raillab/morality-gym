from __future__ import annotations

import textwrap
from typing import Optional, List, Dict, Any, Tuple

import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from morality_gym.environments.core.action import ActionEnum, SubActionEnum
# from morality_gym.environments.core.state import WorldState


# class BaseEvent:
#     def __init__(
#             self,
#             initiated_events: Optional[List[BaseEvent]] = None,
#             caused_events: Optional[List[BaseEvent]] = None,
#             initiated_entities: Optional[List[BaseEntity]] = None,
#             affected_entities: Optional[List[BaseEntity]] = None,
#             state_changes: Optional[Dict[str, Tuple[Any, Any]]] = None,
#             # Not sure about this following one, but can remove if needed
#             state_changed_classifications: Optional[Dict[str, str]] = None,
#             initiated_action_type: Optional[str] = None,  # either direct or indirect
#             initiated_action: Optional[ActionEnum] = None,
#             initiated_sub_action: Optional[SubActionEnum] = None,
#     ):
#         pass

class Event:
    def __init__(
            self,
            timestep: int,
            # Note: Cannot type for BaseEntity since this will cause circular import
            initiated_entities: List[object] = None,
            affected_entities: List[object] = None,

            prev_events: Optional[List[Event]] = None,
            next_events: Optional[List[Event]] = None,

            is_action: bool = False,
            action_descr: Optional[str] = None,
            action: Optional[ActionEnum] = None,
            sub_action: Optional[SubActionEnum] = None,

            is_outcome: bool = False,
            outcome_descr: Optional[str] = None,

            is_causal: bool = False,
            causal_descr: Optional[str] = None,
            causal_sub_descr: Optional[str] = None,

            is_pseudo_event: bool = False,

            state_change_init_entities: Optional[Dict[str, Dict[str, Tuple[Any, Any]]]] = None,
            state_change_affect_entities: Optional[Dict[str, Dict[str, Tuple[Any, Any]]]] = None
    ):
        self.timestep = timestep
        if initiated_entities is None:
            initiated_entities = []
        if affected_entities is None:
            affected_entities = []

        self.initiated_entities = initiated_entities
        self.affected_entities = affected_entities

        if prev_events is None:
            prev_events = []
        if next_events is None:
            next_events = []

        self.prev_events = prev_events
        self.next_events = next_events

        self.is_action = is_action
        self.action_descr = action_descr
        self.action = action
        self.sub_action = sub_action

        self.is_outcome = is_outcome
        self.outcome_descr = outcome_descr

        self.is_causal = is_causal
        self.causal_descr = causal_descr
        self.causal_sub_descr = causal_sub_descr

        self.is_pseudo_event = is_pseudo_event

        self.state_change_init_entities = state_change_init_entities
        self.state_change_affect_entities = state_change_affect_entities


class EventGraph:
    """
    Builds and holds a NetworkX DiGraph representing the causal/sequential
    relationships between Event objects. Includes visualisation method.
    Designed to work with the original Event class definition.
    """
    def __init__(
            self,
            events: List[Event],
            include_connections: bool = False
    ):
        """
        Initialises the EventGraph with a list of events.

        Args:
            events: A list of Event objects from the simulation.
        """
        self.include_connections = include_connections
        self.connections: List[Event] = []
        if self.include_connections:
            # TODO: Test
            tmp_events = []
            for event in events:
                # tmp_events.append(event)
                for prev_event in event.prev_events:
                    if prev_event not in events:
                        tmp_events.append(prev_event)
                for next_event in event.next_events:
                    if next_event not in events:
                        tmp_events.append(next_event)

            self.connections = tmp_events

            events = events + self.connections

        self.events: List[Event] = events
        self.graph: Optional[nx.DiGraph] = None
        # Define colour mapping for nodes
        self.color_map = {
            'action': 'skyblue',
            'outcome': 'lightgreen',
            'causal': 'salmon',
            'unknown': 'lightgrey' # Fallback color
        }


    def build_graph(self) -> nx.DiGraph:
        """
        Constructs the NetworkX DiGraph from the list of events.
        Nodes are Event objects. Edges based on next_events.
        Ignores edges pointing outside the initial event list.
        """
        self.graph = nx.DiGraph()
        event_set = set(self.events) # Use a set for efficient lookup
        for event in self.events:
            self.graph.add_node(event) # Event objects are nodes

        # Add edges based on next_events relationships, only if target is in the graph
        for event in self.events:
            for next_event in event.next_events: # Assumes Event has next_events list
                # Add edge ONLY if the next_event is in the set of initially provided events
                if next_event in event_set:
                    self.graph.add_edge(event, next_event)
                # Silently ignore edges pointing outside the set
        return self.graph

    def get_graph(self) -> Optional[nx.DiGraph]:
        """
        Returns the built graph. Builds it if it hasn't been built yet.
        """
        if not self.events:
            return None
        if self.graph is None:
            self.build_graph()
        return self.graph

    def _format_node_label(self, event: Event) -> str:
        """
        Helper function to create the multi-line label for a node,
        omitting the explicit event type boolean flags.
        """
        # Safely get names, assuming entities have a .name attribute
        try:
            # Accessing event.initiated_entities
            init_names = ', '.join([entity.name for entity in event.initiated_entities]) or 'None'
        except AttributeError:
            init_names = '[Error: Entity has no name]'
        try:
             # Accessing event.affected_entities
            aff_names = ', '.join([entity.name for entity in event.affected_entities]) or 'None'
        except AttributeError:
            aff_names = '[Error: Entity has no name]'

        # --- MODIFICATION START ---
        # Build details based on event type, but don't include the boolean flag itself
        type_details = []
        if event.is_action:
            action_details = []
            action_details.append(f"  Act: {event.action.name if event.action else 'None'}")
            action_details.append(f"  SubAct: {event.sub_action.name if event.sub_action else 'None'}")
            if event.action_descr:
                 action_details.append(f"  Desc: {textwrap.fill(event.action_descr, width=30)}")
            type_details.append("\n".join(action_details))

        if event.is_outcome and event.outcome_descr:
            type_details.append(f"  Desc: {textwrap.fill(event.outcome_descr, width=30)}")

        if event.is_causal and event.causal_descr:
            type_details.append(f"  Desc: {textwrap.fill(event.causal_descr, width=30)}")

        # Join the relevant details for the specific event type
        details_str = "\n".join(type_details)
        # --- MODIFICATION END ---

        label = (
            f"T={event.timestep}\n"
            f"Init: {init_names}\n"
            f"Aff: {aff_names}\n"
            f"{details_str}" # Include only the specific details for the type
        ).strip() # Use strip() to remove potential trailing newline if details_str is empty
        return label

    def vis_graph(
        self,
        img_path: str,
        figsize: Tuple[int, int] = (20, 15), # Adjust figure size
        layout: str = 'spring', # Layout algorithm: 'spring', 'kamada_kawai', 'planar', 'spectral', etc.
        node_size: int = 3000,
        font_size: int = 7,
        k_layout: Optional[float] = None # Parameter for spring_layout
        ):
        """
        Creates and saves a visualisation of the event graph to the specified image path.
        Each node displays detailed information (excluding type flags) and is coloured by event type.

        Args:
            img_path: The full path (including filename and extension, e.g., 'event_graph.png')
                      where the image will be saved.
            figsize: Tuple specifying the figure size in inches.
            layout: The NetworkX layout algorithm to use.
            node_size: Size of the nodes in the plot.
            font_size: Font size for node labels.
            k_layout: Optimal distance between nodes for spring_layout. Adjust for graph density.
        """
        graph = self.get_graph()
        if graph is None or len(graph.nodes) == 0:
            print("Graph is empty or not built. Cannot visualise.")
            return

        # --- Create Labels ---
        labels = {event: self._format_node_label(event) for event in graph.nodes()} # Uses the modified formatter

        # --- Create Node Colors based on Event Type ---
        node_colors = []
        for node_event in graph.nodes():
            # Check boolean flags from the original Event class
            if node_event.is_action:
                node_colors.append(self.color_map['action'])
            elif node_event.is_outcome:
                node_colors.append(self.color_map['outcome'])
            elif node_event.is_causal:
                node_colors.append(self.color_map['causal'])
            else:
                # Fallback based on assumption one is always true
                print(f"Warning: Event at T={node_event.timestep} has no type flag set. Using unknown color.")
                node_colors.append(self.color_map['unknown'])


        # --- Choose Layout ---
        pos = None
        try:
            if layout == 'spring':
                pos = nx.spring_layout(graph, seed=42, k=k_layout, iterations=50)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(graph)
            elif layout == 'planar':
                 try:
                     is_planar, _ = nx.check_planarity(graph)
                     if is_planar:
                         pos = nx.planar_layout(graph)
                     else:
                         print("Warning: Graph is not planar, falling back to spring layout.")
                         pos = nx.spring_layout(graph, seed=42, k=k_layout, iterations=50)
                 except nx.NetworkXException:
                      print("Warning: Planarity check failed, falling back to spring layout.")
                      pos = nx.spring_layout(graph, seed=42, k=k_layout, iterations=50)
            elif layout == 'spectral':
                pos = nx.spectral_layout(graph)
            else:
                print(f"Warning: Unknown layout '{layout}', falling back to spring layout.")
                pos = nx.spring_layout(graph, seed=42, k=k_layout, iterations=50)
        except Exception as e:
            print(f"Error during layout calculation ({layout}): {e}. Falling back to spring layout.")
            pos = nx.spring_layout(graph, seed=42, k=k_layout, iterations=50)


        # --- Draw and Save ---
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.6)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=font_size, font_family='sans-serif')

        # --- Add Legend ---
        legend_handles = [mpatches.Patch(color=self.color_map[event_type], label=event_type.capitalize())
                          for event_type in ['action', 'outcome', 'causal'] if event_type in self.color_map]
        plt.legend(handles=legend_handles, loc='upper right', title="Event Types")


        plt.title("Event Graph Visualisation", fontsize=16)
        plt.axis('off') # Hide axes
        plt.tight_layout()

        try:
            plt.savefig(img_path, format=img_path.split('.')[-1], dpi=300, bbox_inches='tight')
            print(f"Graph visualisation saved to: {img_path}")
        except Exception as e:
            print(f"Error saving graph image to {img_path}: {e}")
        finally:
            plt.close() # Close the figure to free memory
