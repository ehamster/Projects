package com.parkingauthorityticketingsystem;

import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.widget.TextView;

/**
 * Created by yiwu on 10/6/2017.
 */

public class RecentEntriesViewHolder extends RecyclerView.ViewHolder {

    TextView plate_text;
    TextView state_text;
    TextView street_text;
    TextView timeLeft_text;
    public RecentEntriesViewHolder(View itemView) {
        super(itemView);
        plate_text = (TextView) itemView.findViewById(R.id.Plate);
        state_text =(TextView) itemView.findViewById(R.id.State);
        street_text = (TextView) itemView.findViewById(R.id.Street);
        timeLeft_text = (TextView) itemView.findViewById(R.id.TimeLeft);

    }
}
