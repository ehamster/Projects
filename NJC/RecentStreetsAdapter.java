package com.parkingauthorityticketingsystem;

import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.ArrayList;

/**
 * Created by shereen on 11/21/2017.
 */

public class RecentStreetsAdapter extends RecyclerView.Adapter {

    //All methods in this adapter are required for a bare minimum recyclerview adapter
    private int listItemLayout;
    private ArrayList<String> streetList;
    // Constructor of the class
    public RecentStreetsAdapter(int layoutId, ArrayList<String> streetList) {
        listItemLayout = layoutId;
        this.streetList = streetList;
    }

    // get the size of the list
    @Override
    public int getItemCount() {
        return streetList == null ? 0 : streetList.size();
    }


    // specify the row layout file and click for each row
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(listItemLayout, parent, false);
        //RecentStreetsViewHolder myViewHolder = new RecentStreetsViewHolder(view);
        return new SL_ListViewFragment.RecentStreetsViewHolder(view);
    }

    // load data in each row element
    @Override
    public void onBindViewHolder(RecyclerView.ViewHolder holder, final int listPosition) {
       String street = streetList.get(listPosition);

        ((SL_ListViewFragment.RecentStreetsViewHolder)holder).item.setText(street);

//        TextView item = holder.item;
//        item.setText(streetList.get(listPosition));
    }

//    // Static inner class to initialize the views of rows
//    static class RecentStreetsViewHolder extends RecyclerView.RecentStreetsViewHolder implements View.OnClickListener {
//        public TextView item;
//        //static String selected;
//        public RecentStreetsViewHolder(View itemView) {
//            super(itemView);
//            itemView.setOnClickListener(this);
//            item = (TextView) itemView.findViewById(R.id.row_item);
//        }
//        @Override
//        public void onClick(View view) {
//            //Log.d("onclick", "onClick " + getLayoutPosition() + " " + item.getText());
//            System.out.println("onClick " + getLayoutPosition() + " " + item.getText());
//        }
//    }
}
