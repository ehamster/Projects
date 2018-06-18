package com.parkingauthorityticketingsystem;

import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.RecyclerView.Adapter;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.parkingauthorityticketingsystem.entity.Shiftlog;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/**
 * Created by yiwu on 10/6/2017.
 */

public class RecentEntriesAdapter extends RecyclerView.Adapter {
    private List<Shiftlog> shiftlogs;

    public RecentEntriesAdapter(List<Shiftlog> shiftlog) {
        this.shiftlogs = shiftlog;
    }


    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item,parent,false);
       // View view = MainActivity.Viewholder.getMyView();
        return new RecentEntriesViewHolder(view);
    }

    @Override
    public void onBindViewHolder(RecyclerView.ViewHolder holder, final int position) {
        Shiftlog sl = shiftlogs.get(position);
        ((RecentEntriesViewHolder)holder).plate_text.setText(sl.getPlateNumber());
        ((RecentEntriesViewHolder)holder).state_text.setText(sl.getState());
        ((RecentEntriesViewHolder)holder).street_text.setText(sl.getStreet());
        //Left time calculation part
        long timeLeftMilli = 0;
        SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
        Date curDate = new Date(System.currentTimeMillis());
        try {
            curDate = formatter.parse(sl.getFirstObserve());
        } catch (ParseException e) {
            e.printStackTrace();
        }
        timeLeftMilli = sl.getMaxTime()*60*1000 - (System.currentTimeMillis()- curDate.getTime());
        if(timeLeftMilli <= 0){
            ((RecentEntriesViewHolder)holder).timeLeft_text.setText("Violation");
        }
        else{

            ((RecentEntriesViewHolder)holder).timeLeft_text.setText((timeLeftMilli/60/1000)+1+"");
        }


    }

    @Override
    public int getItemCount() {
        return shiftlogs.size();
    }  //delete item
    public Shiftlog removeItem(int position){
        Shiftlog sl = shiftlogs.remove(position); // get item which would be deleted
        notifyItemRemoved(position);
        notifyItemRangeChanged(position,shiftlogs.size());

        return sl;
    }

    public void refresh(List<Shiftlog> shiftlogs){
        this.shiftlogs = shiftlogs;
        notifyDataSetChanged();
    }
}
