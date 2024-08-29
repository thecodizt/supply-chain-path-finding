import streamlit as st
from core import (
    handle_padding,
    convert_list_to_matrix,
    visualize_network_with_paths,
    combinative_greedy_dfs,
    check_if_solution_is_possible,
    handle_transforms,
    generate_random_graph,
)


def main():
    st.title("Optimal Path Detecting with Greedy Cost Minimization")

    st.subheader("Inputs")

    target_demand = st.number_input(
        "Target demand", min_value=0, max_value=100, value=100
    )

    use_sample = st.checkbox(label="Use sample data?", value=False)

    initial_edge_list = None
    capacity_luist = None
    cost_list = None
    num_nodes = None

    nodes_to_avoid = None

    if use_sample:
        inital_edge_list = [(1, 4), (4, 6)]
        capacity_list = [
            (1, 4, 10),
            (1, 5, 10),
            (2, 4, 10),
            (2, 5, 10),
            (3, 4, 10),
            (3, 5, 10),
            (4, 6, 10),
            (4, 7, 10),
            (4, 8, 10),
            (5, 6, 10),
            (5, 7, 10),
            (5, 8, 10),
        ]
        cost_list = [
            (1, 4, 10),
            (1, 5, 10),
            (2, 4, 10),
            (2, 5, 10),
            (3, 4, 10),
            (3, 5, 10),
            (4, 6, 10),
            (4, 7, 10),
            (4, 8, 10),
            (5, 6, 10),
            (5, 7, 10),
            (5, 8, 10),
        ]

        num_nodes = 8
    else:
        num_nodes = st.slider("Number of nodes", min_value=2, max_value=100, value=8)
        inital_edge_list, capacity_list, cost_list = generate_random_graph(num_nodes)

    if num_nodes:
        is_nodes_to_avoid = st.checkbox(label="Use nodes to avoid?", value=False)

        if is_nodes_to_avoid:
            nodes_to_avoid = st.multiselect(
                "Nodes to avoid", options=list(range(1, num_nodes + 1))
            )

    st.subheader("Transforms")

    initial_matrix, capacity_matrix, cost_matrix = handle_transforms(
        initial_edge_list=inital_edge_list,
        capacity_list=capacity_list,
        cost_list=cost_list,
        num_nodes=num_nodes,
    )

    with st.expander("Matrix transformations", expanded=False):

        st.write(initial_matrix)
        st.write(capacity_matrix)
        st.write(cost_matrix)

    st.subheader("Computation")

    is_possible = check_if_solution_is_possible(
        cost_matrix=cost_matrix,
        capacity_matrix=capacity_matrix,
        super_source=0,
        super_target=len(capacity_matrix) - 1,
        target_demand=target_demand,
    )

    if is_possible:
        st.success("Solution is possible!")

        results = combinative_greedy_dfs(
            cost_matrix=cost_matrix,
            capacity_matrix=capacity_matrix,
            super_source=0,
            super_target=len(capacity_matrix) - 1,
            target_demand=target_demand,
            nodes_to_avoid=nodes_to_avoid,
        )

        paths, cost, max_flow = results[0]

        st.subheader("Visualization")

        st.write(f"Paths: {paths}")
        st.write(f"Total Cost: {cost}")
        st.write(f"Max Flow: {max_flow}")
        fig = visualize_network_with_paths(
            capacity_matrix, cost_matrix, paths, max_flow
        )

        st.pyplot(fig)

    else:
        st.error("Solution is not possible!")


if __name__ == "__main__":
    main()
