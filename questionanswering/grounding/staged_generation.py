import logging
from copy import copy
from typing import List

from questionanswering.construction.graph import WithScore
from questionanswering.construction import graph
from questionanswering.construction.graph import SemanticGraph, Edge
from questionanswering.construction import sentence
from questionanswering.datasets import evaluation
from questionanswering.grounding import graph_queries, stages
from questionanswering.models import vectorization as V

MIN_F_SCORE_TO_STOP = 0.9
MAX_ITERATIONS = 1000

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def generate_with_gold(graph_with_scores, gold_answers):
    """
    Generate all possible groundings that produce positive f-score starting with the given ungrounded graph and
    using expand and restrict operations on its denotation.
    自己翻译：通过对给定的未接地图的表示使用扩展和限制操作，来生成所有可能产生正f分的接地图
    
    谷歌翻译：生成所有可能的基础图，从给定的未接地图开始产生正f分，并对其表示使用扩展和限制操作
    参数：graph_with_scores：起始图的元组，应该包含一个标记列表和一个实体列表以及一个零分数
    参数：gold_answers:已编码问题的黄金答案列表
    :param graph_with_scores: a tuple of the starting graph that should contain a list of tokens and a list of entities and a zero score
    :param gold_answers: list of gold answers for the encoded question
    返回值：生成的接地图列表
    :return: a list of generated grounded graphs
    """
    pool = [graph_with_scores]  # pool of possible parses
    if len(gold_answers) == 0 or not any(gold_answers):
        return pool
    positive_graphs, negative_graphs = [], []
    iterations = 0
    while pool \
            and (max(g.scores[2] for g in positive_graphs) if len(positive_graphs) > 0 else 0.0) < MIN_F_SCORE_TO_STOP \
            and iterations < MAX_ITERATIONS:
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))
        master_g_fscore = g.scores[2]
        if master_g_fscore < MIN_F_SCORE_TO_STOP:
            chosen_graphs = []
            f_i = 0
            while f_i < len(stages.ACTIONS) and not chosen_graphs:
                suggested_graphs = stages.ACTIONS[f_i](g[0])
                logger.debug("Suggested graphs: {}".format(suggested_graphs))
                for s_g in suggested_graphs:
                    iterations += 1
                    temp_chosen_graphs, not_chosen_graphs = ground_one_with_gold(s_g, gold_answers, master_g_fscore)
                    negative_graphs += not_chosen_graphs
                    chosen_graphs += temp_chosen_graphs
                f_i += 1
            positive_graphs += chosen_graphs
            logger.debug("Chosen graphs length: {}".format(len(chosen_graphs)))

            if len(chosen_graphs) > 0:
                logger.debug("Extending the pool.")
                pool.extend(chosen_graphs)
                pool = sorted(pool, key=lambda x: (len(x.graph.edges), 1-x.scores[2]), reverse=False)

    negative_graphs = sorted(negative_graphs, key=lambda x: (len(x.graph.edges), -len(x.graph.denotations)), reverse=True)
    positive_graphs = sorted(positive_graphs, key=lambda x: x.scores[2], reverse=True)
    return_graphs = positive_graphs + negative_graphs[:100]
    for g in return_graphs:
        g.graph.denotation_classes = graph_queries.get_graph_groundings(
            stages.with_denotation_class_edge(g.graph))
    logger.debug(f"Iterations {iterations}")
    logger.debug(f"Negative {len(negative_graphs)}")
    if iterations >= MAX_ITERATIONS:
        logger.error(f"Max iterations reached: {iterations}")
    return return_graphs


#？？？
def ground_one_with_gold(s_g, gold_answers, min_fscore):
    grounded_graphs = [apply_grounding(s_g, p) for p in graph_queries.get_graph_groundings(s_g)]
    logger.debug("Number of possible groundings: {}".format(len(grounded_graphs)))
    logger.debug("First one: {}".format(grounded_graphs[:1]))
    i = 0
    chosen_graphs, not_chosen_graphs = [], []
    last_f1 = 0.0
    while i < len(grounded_graphs) and last_f1 < MIN_F_SCORE_TO_STOP:
        s_g = grounded_graphs[i]
        s_g.denotations = graph_queries.get_graph_denotations(s_g)
        i += 1
        retrieved_answers = s_g.denotations

        evaluation_results = evaluation.retrieval_prec_rec_f1(gold_answers, retrieved_answers)
        last_f1 = evaluation_results[2]
        if last_f1 > min_fscore:
            chosen_graphs.append(WithScore(s_g, evaluation_results))
        elif last_f1 < 0.05:
            not_chosen_graphs.append(WithScore(s_g, evaluation_results))
    return chosen_graphs, not_chosen_graphs


def apply_grounding(g: SemanticGraph, grounding) -> SemanticGraph:
    """
    Given a grounding obtained from WikiData apply it to the graph.
    给定一个从WikiData获得的基础，将它应用到图中
    注意：WikiData返回的变量名称很重要，因为它们编码了一些基础功能
    Note: that the variable names returned by WikiData are important as they encode some grounding features.
    参数：g：单个未接地图
    :param g: a single ungrounded graph
    参数：grounding:代表关系和变量基础的字典
    :param grounding: a dictionary representing the grounding of relations and variables
    返回值：接地图
    :return: a grounded graph
    >>> apply_grounding(SemanticGraph([Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q76")]), {'r0v':'P31v'})
    SemanticGraph([Edge(0, ?qvar-P31->Q76)])
    >>> apply_grounding(SemanticGraph([Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, qualifierentityid="Q76")]), {'r0v':'P161v'})
    SemanticGraph([Edge(0, None-P161->?qvar)])
    >>> apply_grounding(SemanticGraph([Edge(qualifierentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q76")]), {'r0v':'P161q'})
    SemanticGraph([Edge(0, None-None->?qvar)])
    >>> apply_grounding(SemanticGraph([Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q76"), Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q5")]), {'r1v':'P39v', 'r0v':'P31v'})
    SemanticGraph([Edge(0, ?qvar-P31->Q76), Edge(1, ?qvar-P39->Q5)])
    >>> apply_grounding(SemanticGraph(), {})
    SemanticGraph([])
    """
    grounded = copy(g)
    for edge in grounded.edges:
        if f"r{edge.edgeid:d}v" in grounding:
            if edge.relationid not in graph_queries.sparql_class_relation:
                relation_id = grounding[f"r{edge.edgeid:d}v"][:-1]
                branch = grounding[f"r{edge.edgeid:d}v"][-1]
                if branch == 'q':
                    edge.qualifierrelationid= relation_id
                    edge.rightentityid, edge.qualifierentityid = edge.qualifierentityid, edge.rightentityid
                else:
                    edge.relationid = relation_id
    return grounded


#？？？
def ground_with_model(input_graphs, s, qa_model, min_score, beam_size=10, verify_with_wikidata=True):
    """
    
    参数：input_graphs: 可供选择的等效图扩展列表
    :param input_graphs: a list of equivalent graph extensions to choose from.
    参数：s:句子
    :param s: sentence
    参数：qa_model:评估图的模型
    :param qa_model: a model to evaluate graphs
    参数：min_score:过滤掉得分低于模型得分的图表
    :param min_score: filter out graphs that receive a score lower than that from the model.
    参数：beam_size:
    :param beam_size: size of the beam
    返回值：一个size = beam_size的选定图列表
    :return: a list of selected graphs with size = beam_size
    """

    logger.debug("Input graphs: {}".format(len(input_graphs)))
    logger.debug("First input one: {}".format(input_graphs[:1]))

    grounded_graphs = [apply_grounding(s_g, p) for s_g in input_graphs for p in graph_queries.get_graph_groundings(s_g, use_wikidata=verify_with_wikidata)]
    grounded_graphs = filter_second_hops(grounded_graphs)
    logger.debug("Number of possible groundings: {}".format(len(grounded_graphs)))
    if len(grounded_graphs) == 0:
        return []

    sentences = []
    for i in range(0, len(grounded_graphs), V.MAX_NEGATIVE_GRAPHS):
        dummy_sentence = sentence.Sentence()
        dummy_sentence.__dict__.update(s.__dict__)
        dummy_sentence.graphs = [WithScore(s_g, (0.0, 0.0, min_score)) for s_g in grounded_graphs[i:i+V.MAX_NEGATIVE_GRAPHS]]
        sentences.append(dummy_sentence)
    if len(sentences) == 0:
        return []
    samples = V.encode_for_model(sentences, qa_model._model.__class__.__name__)
    model_scores = qa_model.predict_batchwise(*samples).view(-1).data

    logger.debug("model_scores: {}".format(model_scores))
    all_chosen_graphs = [WithScore(grounded_graphs[i], (0.0, 0.0, model_scores[i]))
                         for i in range(len(grounded_graphs)) if model_scores[i] > min_score]

    all_chosen_graphs = sorted(all_chosen_graphs, key=lambda x: x[1], reverse=True)
    if len(all_chosen_graphs) > beam_size:
        all_chosen_graphs = all_chosen_graphs[:beam_size]
    logger.debug("Number of chosen groundings: {}".format(len(all_chosen_graphs)))
    return all_chosen_graphs


def filter_second_hops(grounded_graphs: List[SemanticGraph]):
    """
    This methods filters out second hop relations that are already present as first hop relations. Relation direction is
    respected.
    该方法过滤掉已经作为第一跳关系存在的第二跳关系。 关系方向得到尊重？？尊重的意思？
    参数：grounded_graphs:接地图列表
    :param grounded_graphs: list of grounded graphs
    返回值：已过滤的接地图列表
    :return: filtered list of grounded graphs
    >>> g = graph.SemanticGraph(free_entities=[{"type":"NNP", "token_ids":[4], "linkings": [("Q158707", None)]}])
    >>> filter_second_hops([apply_grounding(s_g, p) for s_g in stages.add_entity_and_relation(g, leg_length=1) + stages.add_entity_and_relation(g, leg_length=2, fixed_relations=['P26']) for p in graph_queries.get_graph_groundings(s_g)])
    """

    first_order_relations = {r for g in grounded_graphs for e in g.edges for r in {e.relationid, e.qualifierrelationid}
                             if any(n.startswith("Q") for n in e.nodes() if n) and graph_queries.QUESTION_VAR in e.nodes() and r}
    grounded_graphs = [g for g in grounded_graphs
                       if all((e.relationid not in first_order_relations and e.qualifierrelationid not in first_order_relations)
                              or any(n.startswith("Q") for n in e.nodes() if n) for e in g.edges)]
    return grounded_graphs


def generate_with_model(s, qa_model, beam_size=10):
    pool = [WithScore(s.graphs[0].graph, (0.0, 0.0, 0.0))]  # pool of possible parses
    generated_graphs = []
    iterations = 0

    actions = [
        lambda x: stages.add_entity_and_relation(x, leg_length=1) +
                  stages.add_entity_and_relation(x,
                                                 leg_length=2,
                                                 fixed_relations=stages.LONG_LEG_RELATIONS),
        stages.last_edge_numeric_constraint,
        stages.add_relation
    ]

    while pool and iterations < 100:
        iterations += 1
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))
        master_score = g.scores[2]
        a_i = 0
        chosen_graphs = []
        while a_i < len(actions) and not chosen_graphs:
            suggested_graphs = actions[a_i](g[0])
            suggested_graphs = [s_g for s_g in suggested_graphs if sum(1 for e in s_g.edges
                                if any(n.startswith("Q") for n in e.nodes() if n) and graph_queries.QUESTION_VAR not in e.nodes()) < 2]
            suggested_graphs = [s_g for s_g in suggested_graphs if graph_queries.verify_grounding(s_g)]
            logger.debug("Suggested graphs:{}, {}".format(len(suggested_graphs), suggested_graphs))
            chosen_graphs += ground_with_model(suggested_graphs, s, qa_model, min_score=master_score,
                                               beam_size=beam_size, verify_with_wikidata=True)
            a_i += 1

        logger.debug("Chosen graphs length: {}".format(len(chosen_graphs)))
        if len(chosen_graphs) > 0:
            logger.debug("Extending the pool.")
            pool.extend(chosen_graphs)
            logger.debug("Extending the generated graph set: {}".format(len(chosen_graphs)))
            generated_graphs.extend(chosen_graphs)
    logger.debug("Iterations {}".format(iterations))
    logger.debug("Generated graphs {}".format(len(generated_graphs)))
    generated_graphs = sorted(generated_graphs, key=lambda x: x[1], reverse=True)
    return generated_graphs


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
