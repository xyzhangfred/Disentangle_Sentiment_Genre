def train_Bert(args, train_dataset, model, params_to_update = "ALL", evaluate_during_training = True, eval_dataset = None,\
     pred_genre = False, pred_both = False, multiple_eval_datasets = None, multi_task = False):
    """ Train the model """
    """ 
    Specify which parameters to update. Three strategies, "ALL" means fine-tune the whole model, 
    "Classifier" means keeping all the representation fixed, only train the parameters of the classifier.
    "CLS" means updating the pooling layer and the classifier? 
    """
    logger = logging.getLogger("root")
    final_results = {}
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = min(args.max_steps,  args.num_train_epochs *(len(train_dataloader)* args.gradient_accumulation_steps))
        logging_steps = min(args.logging_steps, t_total//2)
        #args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if params_to_update == "ALL":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        logger.info("Updated params are: %s", "All" )
    elif params_to_update == "Classifier":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)\
                and ( "classifier" in n or "pooler" in n ) ], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)\
                and ( "classifier" in n or "pooler" in n )], 'weight_decay': 0.0}
            ]
        logger.info("Updated params are: %s", [n for n, p in model.named_parameters() if  "classifier" in n or "pooler" in n] )
        

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if not multi_task and not pred_both:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]
            elif pred_both:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         batch[3] + batch[4] * 2}
                outputs = model(**inputs)
                loss = outputs[0]
            else:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels_1':batch[3],  'labels_2':batch[4],}
                outputs = model(**inputs)
                loss = outputs[0]

            if args.n_gpu > 1:
                # breakpoint()    
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and evaluate_during_training and (global_step % logging_steps == 0 or global_step == 1):
                    # Log metrics
                    if multiple_eval_datasets is not None:
                        final_results[global_step] = []
                        for eval_dataset in multiple_eval_datasets:
                            results, ev_loss = evaluate(args = args, eval_dataset = eval_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
                            final_results[global_step].append(results['acc'])
                    else:
                        results, ev_loss = evaluate(args = args, eval_dataset = eval_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
                        logger.info("the %d step :, %s, %s, eval loss : %s, training loss : %s", global_step, results['acc'], results['macro_f1'], ev_loss , tr_loss/logging_steps)
                        final_results[global_step] = (results['acc'])
                        
                        tr_loss = 0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if multi_task:
                        model_to_save = model.module.bert if hasattr(model, 'module') else model.bert 
                    else:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    results, ev_loss = evaluate(args = args, eval_dataset = eval_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
    logger.info("the %d step :, %s, %s", global_step, results['acc'], results['macro_f1'])
    
    logger.warning('########final_results############')
    logger.warning(final_results)
    with open(os.path.join(args.output_dir, "finetune_final_results.json"),"w") as f:
        f.write(json.dumps(final_results))
    final_results = json.load(open(os.path.join(args.output_dir, 'finetune_final_results.json')))
    # plot_results_Bert(final_results, args.output_dir, show = False, pred_both=pred_both)
    return global_step, tr_loss / global_step