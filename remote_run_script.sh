#!/usr/bin/env bash
instance="--instance-ids i-0e47679e23e5ab457"

state=$(aws ec2 describe-instances $instance --query 'Reservations[0].Instances[0].State.Code')

if [ $state -ne 16 ]; then
  if [ $state = 64 ]; then
    echo "Waiting for instance to stop..."
    aws ec2 wait instance-stopped $instance
  fi

  echo "Starting instance..."
  aws ec2 start-instances $instance
  aws ec2 wait instance-running $instance
  aws ec2 wait instance-status-ok $instance
fi

host=$(aws ec2 describe-instances $instance --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
cat $1 | ssh -i ~/.ssh/mykeypair.pem -o StrictHostKeyChecking=accept-new ubuntu@$host
aws ec2 stop-instances $instance
