package com.parkingauthorityticketingsystem;


import android.app.Activity;
import android.app.AlertDialog;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.support.v7.widget.DividerItemDecoration;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.parkingauthorityticketingsystem.database.ShiftlogDBHelper;
import com.parkingauthorityticketingsystem.entity.Shiftlog;
import com.parkingauthorityticketingsystem.utils.DialogGreenFragment;
import com.parkingauthorityticketingsystem.utils.DialogRedFragment;
import com.parkingauthorityticketingsystem.utils.DialogYellowFragment;
import com.parkingauthorityticketingsystem.utils.RecyclerItemClickListener;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


/**
 * A simple {@link Fragment} subclass.
 */
public class SL_ListViewFragment extends Fragment implements DialogRedFragment.DialogFragmentDBEdit,DialogGreenFragment.DialogFragmentDBEdit,
        DialogYellowFragment.DialogFragmentDBEdit{

    //InterfaceRequestLoc1 mCallback_1;
    RequestLocInterface requestLocInterface;

    private ShiftlogDBHelper helper;
    private RecyclerView recyclerViewa, recyclerViewb;
    private ArrayList<String> streets = new ArrayList<>();
    private List<Shiftlog> shiftlogs = new ArrayList<>();
    private RecentEntriesAdapter adapter;
    private RecentStreetsAdapter adapter_a;
    View view;
    String selectedStreet;
    RecentEntriesAdapter recentEntriesAdapter;
    private String location;
    double x_coordinate;
    double y_coordinate;


    public SL_ListViewFragment() {
        // Required empty public constructor
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        Activity activity;
        if(context instanceof Activity){
            activity = (Activity)context;

            // This makes sure that the container activity has implemented
            // the callback interface. If not, it throws an exception
            try {
                requestLocInterface = (RequestLocInterface) activity;
            } catch (ClassCastException e) {
                throw new ClassCastException(activity.toString()
                        + " must implement InterfaceRequestLoc1");
            }
        }
    }

    @Override
    public void onDetach() {
        requestLocInterface = null; // => avoid leaking, thanks @Deepscorn
        super.onDetach();
    }
    public interface InterfaceRequestLoc1 {
        public void requestLocation(String a);
    }
    public void getCoordinates(String info){
        String[] result = info.split(":");
        // System.out.println("getCoordinates: "+result[1]);
        location = result[1];
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        view = inflater.inflate(R.layout.fragment_sl_list_view, container, false);
        helper = new ShiftlogDBHelper(getContext()); // Create Database with name : shiftlog.db
        //First List
        streets = helper.queryStreets();
        final RecentStreetsAdapter recentStreetsAdapter = new RecentStreetsAdapter(R.layout.street_item,streets);
        recyclerViewa = (RecyclerView)view.findViewById(R.id.recyclerViewa);
        recyclerViewa.setLayoutManager(new LinearLayoutManager(view.getContext()));
        recyclerViewa.addItemDecoration(new DividerItemDecoration(view.getContext(),DividerItemDecoration.VERTICAL));
        recyclerViewa.setAdapter(recentStreetsAdapter);
        System.out.println("Streets: "+streets);

        //Second List
        if(streets.size() == 0){

        }
        else
            selectedStreet = streets.get(0);
        secondStreet();

        recyclerViewa.addOnItemTouchListener(
                new RecyclerItemClickListener(getContext(), recyclerViewa ,new RecyclerItemClickListener.OnItemClickListener() {
                    @Override public void onItemClick(View view, int position) {

                        //System.out.println("From inside1: "+position+" "+streets.get(position));
                        selectedStreet = streets.get(position);
                        secondStreet();
                    }

                    @Override public void onLongItemClick(View view, int position) {

                        //System.out.println("From inside2: "+position+" "+streets.get(position));
                        selectedStreet = streets.get(position);
                        secondStreet();
                    }
                })
        );

       /* view.findViewById(R.id.viewMap_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                System.out.println("Change to Map View");
                requestLocInterface.requestLocation("saveLoc1");
                System.out.println("Result: "+location);

                MapViewFragment mapViewFragment = new MapViewFragment();
                Bundle bundle = new Bundle();  //Save current location
                requestLocInterface.requestLocation("saveLoc");
                String []xy = location.split(" ");
                x_coordinate = Double.parseDouble(xy[0]);
                y_coordinate = Double.parseDouble(xy[1]);
                bundle.putDouble("x",x_coordinate);
                bundle.putDouble("y",y_coordinate);
                mapViewFragment.setArguments(bundle);  // transform location from here to mapView
                getActivity().getFragmentManager().beginTransaction().replace(R.id.fragmentContainer,mapViewFragment)
                        .addToBackStack(null).commit();
                Log.i("aa","aa");

            }

        });*/


        return view;
    }

     void secondStreet(){

         shiftlogs = helper.queryOne(selectedStreet);
         System.out.println("Big one: "+shiftlogs);
         recentEntriesAdapter = new RecentEntriesAdapter(shiftlogs);
         recyclerViewb = (RecyclerView)view.findViewById(R.id.recyclerViewb);
         recyclerViewb.setLayoutManager(new LinearLayoutManager(view.getContext()));
         recyclerViewb.setAdapter(recentEntriesAdapter);
         recyclerViewb.addItemDecoration(new DividerItemDecoration(view.getContext(),DividerItemDecoration.VERTICAL));

         recyclerViewb.addOnItemTouchListener(
                 new RecyclerItemClickListener(getContext(), recyclerViewa ,new RecyclerItemClickListener.OnItemClickListener() {
                     @Override public void onItemClick(View view, int position) {
                        // System.out.println("oooFrom inside1: "+position);
                         doCheck(position);
                     }

                     @Override public void onLongItemClick(View view, int position) {
                         //System.out.println("oooFrom inside2: "+position);
                     }
                 })
         );

     }

     void doCheck(int pos){

         Shiftlog sl = shiftlogs.get(pos);
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
         timeLeftMilli = timeLeftMilli / 60 / 1000;
         if(timeLeftMilli > 0) {
             timeLeftMilli ++;
         }
         if(timeLeftMilli >= 6 ) {
             showGreenDialog(sl,timeLeftMilli);
         }
         else if (timeLeftMilli < 6 && timeLeftMilli > 0){
             showYellowDialog(sl,timeLeftMilli);
         }
         else{
             showRedDialog(sl,timeLeftMilli);
         }

     }

    private void showRedDialog(Shiftlog sl, double timeLeftMilli) {
        FragmentTransaction mFragTransaction = getFragmentManager().beginTransaction();
        Fragment fragment =  getFragmentManager().findFragmentByTag("dialogFragmentRed");
        if(fragment!=null){
            //Avoid display dialog twice
            mFragTransaction.remove(fragment);
        }
        DialogRedFragment dialogFragment =new DialogRedFragment().newInstance(sl,timeLeftMilli);
        dialogFragment.setDialogFragmentDBEdit(SL_ListViewFragment.this);
        dialogFragment.show(mFragTransaction,"dialogFragmentRed");


    }

    private void showGreenDialog(Shiftlog sl,double timeLeftMilli) {
        FragmentTransaction mFragTransaction = getFragmentManager().beginTransaction();
        Fragment fragment =  getFragmentManager().findFragmentByTag("dialogFragment");
        if(fragment!=null){
            //Avoid display dialog twice
            mFragTransaction.remove(fragment);
        }
        DialogGreenFragment dialogFragment =DialogGreenFragment.newInstance(sl,timeLeftMilli);
        dialogFragment.setDialogFragmentDBEdit(SL_ListViewFragment.this);
        dialogFragment.show(mFragTransaction,"dialogFragment");
    }
    private void showYellowDialog(Shiftlog sl,double timeLeftMilli) {
        FragmentTransaction mFragTransaction = getFragmentManager().beginTransaction();
        Fragment fragment =  getFragmentManager().findFragmentByTag("dialogFragmentYellow");
        if(fragment!=null){
            //Avoid display dialog twice
            mFragTransaction.remove(fragment);
        }
        DialogYellowFragment dialogFragment = DialogYellowFragment.newInstance(sl,timeLeftMilli);
        dialogFragment.setDialogFragmentDBEdit(SL_ListViewFragment.this);
        dialogFragment.show(mFragTransaction,"dialogFragmentYellow");
    }
    // Static inner class to initialize the views of rows
    static class RecentStreetsViewHolder extends RecyclerView.ViewHolder /*implements View.OnClickListener */{
        public TextView item;
        //static String selected;
        public RecentStreetsViewHolder(View itemView) {
            super(itemView);
            //itemView.setOnClickListener(this);
            item = (TextView) itemView.findViewById(R.id.row_item);
        }
//        @Override
//        public void onClick(View view) {
//            //Log.d("onclick", "onClick " + getLayoutPosition() + " " + item.getText());
//            System.out.println("onClick " + getLayoutPosition() + " " + item.getText());
//            selectedStreet=item.getText().toString();
//            //secondStreet();
//        }
    }
    @Override
    public void remove(final int id) {  //rewrite remove() in interface, in DialogRedFragment use this method
        helper = new ShiftlogDBHelper(getContext());
        new AlertDialog.Builder(getContext()).setTitle("Warning").setMessage("Are you sure to remove ?")
                .setNegativeButton("Confirm", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        helper.deleteRecord(id);
                        for(Shiftlog sl : shiftlogs) {
                            if (sl.getId() == id) {
                                shiftlogs.remove(sl);
                                secondStreet();
                                break;
                            }
                        }
                    }
                }).setPositiveButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

            }
        }).create().show();


    }
    public void ticket(Shiftlog sl) {
      /*  helper.deleteRecord(id);
        googleMap.clear();
        for(Shiftlog sl : shiftlogs) {
            if (sl.getId() == id) {
                shiftlogs.remove(sl);
                break;
            }
        }
        DrawUtil.drawMarkers(shiftlogs,googleMap);*/
    }


}
