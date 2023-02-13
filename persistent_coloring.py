                    padded_bboxes = team_classificator.pad_detections(bboxes, max_width, max_height)
                    labels = team_classificator.classify_persons_into_categories(padded_bboxes, max_height, max_width)
                    assert len(labels) == len(frame_ids)
                    
                    sorted_labels = list(dict(sorted(Counter(labels).items(), key=lambda item: item[1])).keys())
                    # Initialize teams
                    mapping = dict()
                    mapping[sorted_labels[0]] = (sorted_labels[0], YELLOW_COLOR)
                    founded_correct = True
                    if team1 is None or team2 is None:
                        team1 = Team()
                        team1.color = DARK_BLUE_COLOR
                        team1.label = sorted_labels[1]
                        team2 = Team()
                        team2.color = DARK_RED_COLOR
                        team2.label = sorted_labels[2]
                    else:  # there are already some players             
                        k = 0
                        founded_correct = False
                        while labels[k] == sorted_labels[0]:  # find me the player that is not of the smallest class type
                            if (labels[k] == team1.label and ids[k] in team1.players and ids[k] not in team2.players) or (
                                labels[k] == team2.label and ids[k] in team2.players and ids[k] not in team1.players
                            ):
                                founded_correct = True
                                break                        
                            k += 1
                    if founded_correct:
                        mapping[sorted_labels[1]] = (team1.label, team1.color)
                        mapping[sorted_labels[2]] = (team2.label, team2.color)
                    else:
                        mapping[sorted_labels[1]] = (team2.label, team2.color)
                        mapping[sorted_labels[2]] = (team1.label, team1.color)
                                                
                    # Output to MOT format
                    with open(txt_path + '.txt', 'a') as f:
                        for j in range(len(frame_ids)):
                            add_bounding_box_to_image(clss[j], ids[j], mapping[labels[j]][1] , hide_labels, names, hide_conf, hide_class, confs[j], annotator, bb_infos[j], 
                                                      save_trajectories, tracking_method, qs[j], tracker_list, im0, i)
                            f.write(('%g ' * 11 + '\n') % (frame_ids[j], ids[j], mapping[labels[j]][0], bbox_lefts[j],  # MOT format
                                                       bbox_tops[j], bbox_ws[j], bbox_hs[j], -1, -1, -1, i))
                            if mapping[labels[j]][0] == team1.label:
                                team1.players.add(ids[j])
                            elif mapping[labels[j]][0] == team2.label:
                                team2.players.add(ids[j])
